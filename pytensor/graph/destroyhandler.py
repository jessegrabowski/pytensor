"""
Classes and functions for validating graphs that contain view
and inplace operations.

"""

import itertools
from collections import deque

from pytensor.configdefaults import config
from pytensor.graph.basic import Constant
from pytensor.graph.features import AlreadyThere, Bookkeeper
from pytensor.graph.fg import Output
from pytensor.graph.utils import InconsistencyError
from pytensor.misc.ordered_set import OrderedSet


class ProtocolError(Exception):
    """
    Raised when FunctionGraph calls DestroyHandler callbacks in
    an invalid way, for example, pruning or changing a node that has
    never been imported.

    """


def _contains_cycle(fgraph, orderings):
    """
    Function to check if the given graph contains a cycle

    Parameters
    ----------
    fgraph
        The FunctionGraph to check for cycles.
    orderings
        Dictionary specifying extra dependencies besides those encoded in
        Variable.owner / Apply.inputs.

        If orderings[my_apply] == dependencies, then my_apply is an Apply
        instance, dependencies is a set of Apply instances, and every member
        of dependencies must be executed before my_apply.

        The dependencies are typically used to prevent
        inplace apply nodes from destroying their input before
        other apply nodes with the same input access it.

    Returns
    -------
    bool
        True if the graph contains a cycle, False otherwise.

    """
    # These are lists of Variable instances
    outputs = fgraph.outputs

    # this is hard-coded reimplementation of functions from graph.py
    # reason: go faster, prepare for port to C.
    # specifically, it could be replaced with a wrapper
    # around graph.io_toposort that returns True iff io_toposort raises
    # a ValueError containing the substring 'cycle'.
    # This implementation is optimized for the destroyhandler and runs
    # slightly faster than io_toposort.

    # this is performance-critical code. it is the largest single-function
    # bottleneck when compiling large graphs.
    assert isinstance(outputs, tuple | list | deque)

    # TODO: For more speed - use a defaultdict for the orderings
    # (defaultdict runs faster than dict in the case where the key
    # is not in the dictionary, at least in CPython)

    # IG: I tried converting parent_counts to use an id for the key,
    # so that the dict would do reference counting on its keys.
    # This caused a slowdown.
    # Separate benchmark tests showed that calling id is about
    # half as expensive as a dictionary access, and that the
    # dictionary also runs slower when storing ids than when
    # storing objects.

    # dict mapping an Apply or Variable instance to the number
    # of its parents (including parents imposed by orderings)
    # that haven't been visited yet
    parent_counts = {}
    # dict mapping an Apply or Variable instance to its children
    node_to_children = {}

    # visitable: A container holding all Variable and Apply instances
    # that can currently be visited according to the graph topology
    # (ie, whose parents have already been visited)
    # TODO: visitable is a fifo_queue. could this run faster if we
    # implement it as a stack rather than a deque?
    # TODO: visitable need not be a fifo_queue, any kind of container
    # that we can throw things into and take things out of quickly will
    # work. is there another kind of container that could run faster?
    # we don't care about the traversal order here as much as we do
    # in io_toposort because we aren't trying to generate an ordering
    # on the nodes
    visitable = deque()

    # IG: visitable could in principle be initialized to fgraph.inputs
    #     + fgraph.orphans... if there were an fgraph.orphans structure.
    #     I tried making one and maintaining it caused a huge slowdown.
    #     This may be because I made it a list, so it would have a
    #     deterministic iteration order, in hopes of using it to speed
    #     up toposort as well.
    #     I think since we need to scan through all variables and nodes
    #     to make parent_counts anyway, it's cheap enough to always
    #     detect orphans at cycle detection / toposort time

    # Pass through all the nodes to build visitable, parent_count, and
    # node_to_children
    for var in fgraph.variables:
        # this is faster than calling get_parents
        owner = var.owner
        # variables don't appear in orderings, so we don't need to worry
        # about that here
        if owner:
            # insert node in node_to_children[r]
            # (if r is not already in node_to_children,
            # initialize it to [])
            node_to_children.setdefault(owner, []).append(var)
            parent_counts[var] = 1
        else:
            visitable.append(var)
            parent_counts[var] = 0

    for a_n in fgraph.apply_nodes:
        parents = list(a_n.inputs)
        # This is faster than conditionally extending
        # IG: I tried using a shared empty_list = [] constructed
        # outside of the for loop to avoid constructing multiple
        # lists, but this was not any faster.
        parents.extend(orderings.get(a_n, []))

        if parents:
            for parent in parents:
                # insert node in node_to_children[r]
                # (if r is not already in node_to_children,
                # initialize it to [])
                node_to_children.setdefault(parent, []).append(a_n)
            parent_counts[a_n] = len(parents)
        else:
            # an Apply with no inputs would be a weird case, but I'm
            # not sure we forbid it
            visitable.append(a_n)
            parent_counts[a_n] = 0

    # at this point,
    # parent_counts.keys() == fgraph.apply_nodes + fgraph.variables

    # Now we actually check for cycles
    # As long as there are nodes that can be visited while respecting
    # the topology, we keep visiting nodes
    # If we run out of visitable nodes and we haven't visited all nodes,
    # then there was a cycle. It blocked the traversal because some
    # node couldn't be visited until one of its descendants had been
    # visited too.
    # This is a standard cycle detection algorithm.

    visited = 0
    while visitable:
        # Since each node is inserted into the visitable queue exactly
        # once, it comes out of the queue exactly once
        # That means we can decrement its children's unvisited parent count
        # and increment the visited node count without double-counting
        node = visitable.popleft()
        visited += 1
        for client in node_to_children.get(node, []):
            parent_counts[client] -= 1
            # If all of a node's parents have been visited,
            # it may now be visited too
            if not parent_counts[client]:
                visitable.append(client)

    return visited != len(parent_counts)


def _build_droot_impact(destroy_handler):
    droot = {}  # destroyed view + nonview variables -> foundation
    impact = {}  # destroyed nonview variable -> it + all views of it
    root_destroyer = {}  # root -> destroyer apply

    for app in destroy_handler.destroyers:
        for input_idx_list in app.op.destroy_map.values():
            if len(input_idx_list) != 1:
                raise NotImplementedError()
            input_idx = input_idx_list[0]
            input = app.inputs[input_idx]

            # Find non-view variable which is ultimately viewed by input.
            view_i = destroy_handler.view_i
            _r = input
            while _r is not None:
                r = _r
                _r = view_i.get(r)
            input_root = r

            if input_root in droot:
                raise InconsistencyError(f"Multiple destroyers of {input_root}")
            droot[input_root] = input_root
            root_destroyer[input_root] = app

            # The code here add all the variables that are views of r into
            # an OrderedSet input_impact
            input_impact = OrderedSet()

            q = deque()
            q.append(input_root)
            while len(q) > 0:
                v = q.popleft()
                for n in destroy_handler.view_o.get(v, []):
                    input_impact.add(n)
                    q.append(n)

            for v in input_impact:
                assert v not in droot
                droot[v] = input_root

            impact[input_root] = input_impact
            impact[input_root].add(input_root)

    return droot, impact, root_destroyer


def inplace_candidates(fgraph, inputs, protected_inputs=None):
    """
    Return the variables in inputs that are possible candidate for as inputs of
    inplace operation.

    Parameters
    ----------
    inputs : list
        Inputs Variable that you want to use as inplace destination.

    """
    if protected_inputs is None:
        from pytensor.compile.function.types import Supervisor

        protected_inputs = set(
            itertools.chain.from_iterable(
                f.protected for f in fgraph._features if isinstance(f, Supervisor)
            )
        )
        protected_inputs.update(fgraph.outputs)

    has_destroyers = fgraph.has_destroyers
    view_i = fgraph.destroy_handler.view_i
    candidate_roots = {}
    candidate_inputs = []
    for inp in inputs:
        if isinstance(inp, Constant):
            # Can't inplace on constants.
            continue

        # Find the root of the view chain, and while traversing check if it passes on any protected inputs.
        view_of_protected = False
        root = inp
        try:
            while True:
                if root in protected_inputs:
                    view_of_protected = True
                root = view_i[root]
        except KeyError:
            pass

        if root in candidate_roots:
            # Another input views on the same root, we can't destroy either
            if (invalid_candidate := candidate_roots[root]) is not None:
                # Invalidate the previous candidate
                candidate_inputs.remove(invalid_candidate)
            candidate_roots[root] = None
        elif not view_of_protected and not has_destroyers([inp]):
            candidate_inputs.append(inp)
            candidate_roots[root] = inp
        else:
            candidate_roots[root] = None

    return candidate_inputs


class DestroyHandler(Bookkeeper):
    """
    The DestroyHandler class detects when a graph is impossible to evaluate
    because of aliasing and destructive operations.

    Several data structures are used to do this.

    An Op can use its view_map property to declare that an output may be
    aliased to an input. If that output is destroyed, the input is also
    considered to be destroyed. The view_maps of several Ops can feed into
    one another and form a directed graph. The consequence of destroying any
    variable in such a graph is that all variables in the graph must be
    considered to be destroyed, because they could all be referring to the
    same underlying storage.

    In the current implementation, that graph is a tree, and the root of that
    tree is called the foundation.

    TODO: why "in the current implementation" ? is there another implementation
          planned?
    TODO: why is the graph a tree? isn't it possible that one variable could
          be aliased to many variables? for example, don't switch and ifelse
          have to do this?

    The original DestroyHandler (if 0'ed out above) computed several data
    structures from scratch each time it was asked to validate the graph.
    Because this happens potentially thousands of times and each graph to
    validate is extremely similar to the previous one, computing the
    data structures from scratch repeatedly was wasteful and resulted in
    high compile times for large graphs.

    This implementation computes the data structures once at initialization
    and then incrementally updates them.

    It is a work in progress. The following data structures have been
    converted to use the incremental strategy:
        <none>

    The following data structures remain to be converted:
        <unknown>

    """

    pickle_rm_attr = ["destroyers", "has_destroyers"]

    def __init__(self, do_imports_on_attach=True, algo=None):
        self.fgraph = None
        self.do_imports_on_attach = do_imports_on_attach

        """
        Maps every variable in the graph to its "foundation" (deepest
        ancestor in view chain).
        TODO: change name to var_to_vroot.

        """
        self.droot = {}

        """
        Maps a variable to all variables that are indirect or direct views of it
        (including itself) essentially the inverse of droot.
        TODO: do all variables appear in this dict, or only those that are
              foundations?
        TODO: do only destroyed variables go in here? one old docstring said so.
        TODO: rename to x_to_views after reverse engineering what x is

        """
        self.impact = {}

        """
        If a var is destroyed, then this dict will map
        droot[var] to the apply node that destroyed var
        TODO: rename to vroot_to_destroyer

        """
        self.root_destroyer = {}
        if algo is None:
            algo = config.cycle_detection
        self.algo = algo
        self.fail_validate = {}

    def clone(self):
        return type(self)(self.do_imports_on_attach, self.algo)

    def on_attach(self, fgraph):
        """
        When attaching to a new fgraph, check that
            1) This DestroyHandler wasn't already attached to some fgraph
               (its data structures are only set up to serve one).
            2) The FunctionGraph doesn't already have a DestroyHandler.
               This would result in it validating everything twice, causing
               compilation to be slower.

        Give the FunctionGraph instance:
            1) A new method "destroyers(var)"
               TODO: what does this do exactly?
            2) A new attribute, "destroy_handler"
        TODO: WRITEME: what does this do besides the checks?

        """

        if any(hasattr(fgraph, attr) for attr in ("destroyers", "destroy_handler")):
            raise AlreadyThere("DestroyHandler feature is already present")

        if self.fgraph is not None and self.fgraph != fgraph:
            raise Exception(
                "A DestroyHandler instance can only serve one FunctionGraph"
            )

        # Annotate the FunctionGraph #
        self.unpickle(fgraph)
        fgraph.destroy_handler = self

        self.fgraph = fgraph
        self.destroyers = (
            OrderedSet()
        )  # set of Apply instances with non-null destroy_map
        self.view_i = {}  # variable -> variable used in calculation
        self.view_o = {}  # variable -> set of variables that use this one as a direct input
        # clients: how many times does an apply use a given variable
        self.clients = {}  # variable -> apply -> ninputs
        self.stale_droot = True

        self.debug_all_apps = set()
        if self.do_imports_on_attach:
            Bookkeeper.on_attach(self, fgraph)

    def unpickle(self, fgraph):
        def get_destroyers_of(r):
            droot, _, root_destroyer = self.refresh_droot_impact()
            try:
                return [root_destroyer[droot[r]]]
            except Exception:
                return []

        fgraph.destroyers = get_destroyers_of

        def has_destroyers(protected_list):
            if self.algo != "fast":
                droot, _, root_destroyer = self.refresh_droot_impact()
                for protected_var in protected_list:
                    try:
                        root_destroyer[droot[protected_var]]
                        return True
                    except KeyError:
                        pass
                return False

            def recursive_destroys_finder(protected_var):
                # protected_var is the idx'th input of app.
                for app, idx in fgraph.clients[protected_var]:
                    destroy_maps = app.op.destroy_map.values()
                    # If True means that the apply node, destroys the protected_var.
                    if idx in [dmap for sublist in destroy_maps for dmap in sublist]:
                        return True
                    for var_idx in app.op.view_map:
                        if idx in app.op.view_map[var_idx]:
                            # We need to recursively check the destroy_map of all the
                            # outputs that we have a view_map on.
                            if recursive_destroys_finder(app.outputs[var_idx]):
                                return True
                return False

            for protected_var in protected_list:
                if recursive_destroys_finder(protected_var):
                    return True
            return False

        fgraph.has_destroyers = has_destroyers

    def refresh_droot_impact(self):
        """
        Makes sure self.droot, self.impact, and self.root_destroyer are up to
        date, and returns them (see docstrings for these properties above).

        """
        if self.stale_droot:
            self.droot, self.impact, self.root_destroyer = _build_droot_impact(self)
            self.stale_droot = False
        return self.droot, self.impact, self.root_destroyer

    def on_detach(self, fgraph):
        if fgraph is not self.fgraph:
            raise Exception("detaching wrong fgraph", fgraph)
        del self.destroyers
        del self.view_i
        del self.view_o
        del self.clients
        del self.stale_droot
        assert self.fgraph.destroyer_handler is self
        delattr(self.fgraph, "destroyers")
        delattr(self.fgraph, "has_destroyers")
        delattr(self.fgraph, "destroy_handler")
        self.fgraph = None

    def fast_destroy(self, fgraph, app, reason):
        """
        Do the check for only 1 level.

        For now:
        - Destroyed variables can have only 1 clients.
        - Allow view to have multiple clients.
        - Allow sequence of view.
        - But don't allow to destroy view
        """
        dm = app.op.destroy_map
        if not dm:
            return
        inputs = set(
            itertools.chain.from_iterable(dm.values())
        )  # list of app's destroyed inputs
        for inp_idx in inputs:
            inp = app.inputs[inp_idx]
            if getattr(inp.tag, "indestructible", False) or isinstance(inp, Constant):
                self.fail_validate[app] = InconsistencyError(
                    f"Attempting to destroy indestructible variables: {inp}"
                )
            elif len(fgraph.clients[inp]) > 1:
                self.fail_validate[app] = InconsistencyError(
                    "Destroyed variable has more than one client. " + str(reason)
                )
            elif inp.owner:
                app2 = inp.owner
                inp_idx2 = app2.outputs.index(inp)
                v = app2.op.view_map
                d = app2.op.destroy_map
                if v:
                    v = v.get(inp_idx2, [])
                    if len(v) > 0:
                        self.fail_validate[app] = InconsistencyError(
                            "Destroyed variable has view_map. " + str(reason)
                        )
                elif d:
                    d = d.get(inp_idx2, [])
                    if len(d) > 0:
                        self.fail_validate[app] = InconsistencyError(
                            "Destroyed variable has destroy_map. " + str(reason)
                        )

                # These 2 assertions are commented since this function is called so many times
                # but they should be true.
                # assert len(v) <= 1
                # assert len(d) <= 1

    def on_import(self, fgraph, app, reason):
        """
        Add Apply instance to set which must be computed.

        """
        if app in self.debug_all_apps:
            raise ProtocolError("double import")
        self.debug_all_apps.add(app)
        # print 'DH IMPORT', app, id(app), id(self), len(self.debug_all_apps)

        # If it's a destructive op, add it to our watch list
        dmap = app.op.destroy_map
        vmap = app.op.view_map
        if dmap:
            self.destroyers.add(app)
            if self.algo == "fast":
                self.fast_destroy(fgraph, app, reason)

        # add this symbol to the forward and backward maps
        for o_idx, i_idx_list in vmap.items():
            if len(i_idx_list) > 1:
                raise NotImplementedError(
                    "destroying this output invalidates multiple inputs", (app.op)
                )
            o = app.outputs[o_idx]
            i = app.inputs[i_idx_list[0]]
            self.view_i[o] = i
            self.view_o.setdefault(i, OrderedSet()).add(o)

        # update self.clients
        for i, input in enumerate(app.inputs):
            self.clients.setdefault(input, {}).setdefault(app, 0)
            self.clients[input][app] += 1

        for i, output in enumerate(app.outputs):
            self.clients.setdefault(output, {})

        self.stale_droot = True

    def on_prune(self, fgraph, app, reason):
        """
        Remove Apply instance from set which must be computed.

        """
        if app not in self.debug_all_apps:
            raise ProtocolError("prune without import")
        self.debug_all_apps.remove(app)

        # UPDATE self.clients
        for input in set(app.inputs):
            del self.clients[input][app]

        if app.op.destroy_map:
            self.destroyers.remove(app)

        # Note: leaving empty client dictionaries in the struct.
        # Why? It's a pain to remove them. I think they aren't doing any harm, they will be
        # deleted on_detach().

        # UPDATE self.view_i, self.view_o
        for o_idx, i_idx_list in app.op.view_map.items():
            if len(i_idx_list) > 1:
                # destroying this output invalidates multiple inputs
                raise NotImplementedError()
            o = app.outputs[o_idx]
            i = app.inputs[i_idx_list[0]]

            del self.view_i[o]

            self.view_o[i].remove(o)
            if not self.view_o[i]:
                del self.view_o[i]

        self.stale_droot = True
        if app in self.fail_validate:
            del self.fail_validate[app]

    def on_change_input(self, fgraph, app, i, old_r, new_r, reason):
        """
        app.inputs[i] changed from old_r to new_r.

        """
        if isinstance(app.op, Output):
            # app == 'output' is special key that means FunctionGraph is redefining which nodes are being
            # considered 'outputs' of the graph.
            pass
        else:
            if app not in self.debug_all_apps:
                raise ProtocolError("change without import")

            # UPDATE self.clients
            self.clients[old_r][app] -= 1
            if self.clients[old_r][app] == 0:
                del self.clients[old_r][app]

            self.clients.setdefault(new_r, {}).setdefault(app, 0)
            self.clients[new_r][app] += 1

            # UPDATE self.view_i, self.view_o
            for o_idx, i_idx_list in app.op.view_map.items():
                if len(i_idx_list) > 1:
                    # destroying this output invalidates multiple inputs
                    raise NotImplementedError()
                i_idx = i_idx_list[0]
                output = app.outputs[o_idx]
                if i_idx == i:
                    if app.inputs[i_idx] is not new_r:
                        raise ProtocolError("wrong new_r on change")

                    self.view_i[output] = new_r

                    self.view_o[old_r].remove(output)
                    if not self.view_o[old_r]:
                        del self.view_o[old_r]

                    self.view_o.setdefault(new_r, OrderedSet()).add(output)

            if self.algo == "fast":
                if app in self.fail_validate:
                    del self.fail_validate[app]
                self.fast_destroy(fgraph, app, reason)
        self.stale_droot = True

    def validate(self, fgraph):
        """
        Return None.

        Raise InconsistencyError when
        a) orderings() raises an error
        b) orderings cannot be topologically sorted.

        """
        if self.destroyers:
            if self.algo == "fast":
                if self.fail_validate:
                    app_err_pairs = self.fail_validate
                    self.fail_validate = {}
                    # self.fail_validate can only be a hint that maybe/probably
                    # there is a cycle.This is because inside replace() we could
                    # record many reasons to not accept a change, but we don't
                    # know which one will fail first inside validate(). Thus,the
                    # graph might have already changed when we raise the
                    # self.fail_validate error. So before raising the error, we
                    # double check here.
                    for app in app_err_pairs:
                        if app in fgraph.apply_nodes:
                            self.fast_destroy(fgraph, app, "validate")
                    if self.fail_validate:
                        self.fail_validate = app_err_pairs
                        raise app_err_pairs[app]
            else:
                ords = self.orderings(fgraph, ordered=False)
                if _contains_cycle(fgraph, ords):
                    raise InconsistencyError("Dependency graph contains cycles")
        else:
            # James's Conjecture:
            # If there are no destructive ops, then there can be no cycles.

            # FB: This isn't always True. It can happened that
            # optimization introduce node that depend on itself. This
            # is very rare and should not happen in general. It will be
            # caught later. The error will be far from the source. But
            # doing this conjecture should speed up compilation most of
            # the time. The user should create such dependency except
            # if he mess too much with the internal.
            pass
        return True

    def orderings(self, fgraph, ordered=True):
        """
        Return orderings induced by destructive operations.

        Raise InconsistencyError when
        a) attempting to destroy indestructable variable, or
        b) attempting to destroy a value multiple times, or
        c) an Apply destroys (illegally) one of its own inputs by aliasing

        """
        set_type = OrderedSet if ordered else set
        rval = {}

        if self.destroyers:
            # BUILD DATA STRUCTURES
            # CHECK for multiple destructions during construction of variables

            droot, impact, __ignore = self.refresh_droot_impact()

            # check for destruction of constants
            illegal_destroy = [
                r
                for r in droot
                if getattr(r.tag, "indestructible", False) or isinstance(r, Constant)
            ]
            if illegal_destroy:
                raise InconsistencyError(
                    f"Attempting to destroy indestructible variables: {illegal_destroy}"
                )

            # add destroyed variable clients as computational dependencies
            for app in self.destroyers:
                # keep track of clients that should run before the current Apply
                root_clients = set_type()
                # for each destroyed input...
                for input_idx_list in app.op.destroy_map.values():
                    destroyed_idx = input_idx_list[0]
                    destroyed_variable = app.inputs[destroyed_idx]
                    root = droot[destroyed_variable]
                    root_impact = impact[root]
                    # we generally want to put all clients of things which depend on root
                    # as pre-requisites of app.
                    # But, app is itself one such client!
                    # App will always be a client of the node we're destroying
                    # (destroyed_variable, but the tricky thing is when it is also a client of
                    # *another variable* viewing on the root.  Generally this is illegal, (e.g.,
                    # add_inplace(x, x.T).  In some special cases though, the in-place op will
                    # actually be able to work properly with multiple destroyed inputs (e.g,
                    # add_inplace(x, x).  An Op that can still work in this case should declare
                    # so via the 'destroyhandler_tolerate_same' attribute or
                    # 'destroyhandler_tolerate_aliased' attribute.
                    #
                    # destroyhandler_tolerate_same should be a list of pairs of the form
                    # [(idx0, idx1), (idx0, idx2), ...]
                    # The first element of each pair is the input index of a destroyed
                    # variable.
                    # The second element of each pair is the index of a different input where
                    # we will permit exactly the same variable to appear.
                    # For example, add_inplace.tolerate_same might be [(0,1)] if the destroyed
                    # input is also allowed to appear as the second argument.
                    #
                    # destroyhandler_tolerate_aliased is the same sort of list of
                    # pairs.
                    # op.destroyhandler_tolerate_aliased = [(idx0, idx1)] tells the
                    # destroyhandler to IGNORE an aliasing between a destroyed
                    # input idx0 and another input idx1.
                    # This is generally a bad idea, but it is safe in some
                    # cases, such as
                    # - the op reads from the aliased idx1 before modifying idx0
                    # - the idx0 and idx1 are guaranteed not to overlap (e.g.
                    #   they are pointed at different rows of a matrix).
                    #

                    # CHECK FOR INPUT ALIASING
                    # OPT: pre-compute this on import
                    tolerate_same = getattr(app.op, "destroyhandler_tolerate_same", [])
                    assert isinstance(tolerate_same, list)
                    tolerated = {
                        idx1 for idx0, idx1 in tolerate_same if idx0 == destroyed_idx
                    }
                    tolerated.add(destroyed_idx)
                    tolerate_aliased = getattr(
                        app.op, "destroyhandler_tolerate_aliased", []
                    )
                    assert isinstance(tolerate_aliased, list)
                    ignored = {
                        idx1 for idx0, idx1 in tolerate_aliased if idx0 == destroyed_idx
                    }
                    for i, input in enumerate(app.inputs):
                        if i in ignored:
                            continue
                        if input in root_impact and (
                            i not in tolerated or input is not destroyed_variable
                        ):
                            raise InconsistencyError(
                                f"Input aliasing: {app} ({destroyed_idx}, {i})"
                            )

                    # add the rule: app must be preceded by all other Apply instances that
                    # depend on destroyed_input
                    for r in root_impact:
                        assert not [a for a, c in self.clients[r].items() if not c]
                        root_clients.update(
                            [a for a, c in self.clients[r].items() if c]
                        )

                # app itself is a client of the destroyed inputs,
                # but should not run before itself
                root_clients.remove(app)
                if root_clients:
                    rval[app] = root_clients

        return rval
