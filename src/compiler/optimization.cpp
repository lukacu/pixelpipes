

#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <valarray>
#include <algorithm>

#include "optimization.hpp"
#include "../debug.h"

namespace pixelpipes
{

    DNF::DNF() = default;
    DNF::~DNF() = default;

    DNF::DNF(std::vector<std::vector<bool>> clauses) : _clauses(clauses)
    {
        _size = 0;

        for (auto x : _clauses)
        {
            _size += x.size();
        }
    }

    size_t DNF::size() const
    {
        return _size;
    }

    bool DNF::compute(TokenList values) const
    {
        VERIFY(values.size() == _size, "Unexpected number of inputs to DNF clause");

        auto v = values.begin();

        for (size_t i = 0; i < _clauses.size(); i++)
        {
            auto offset = v;
            bool result = true;
            for (bool negate : _clauses[i])
            {
                result &= (*v) && (extract<bool>(*v) != negate);
                if (!result)
                {
                    v = offset + _clauses[i].size();
                    break;
                }
                else
                    v++;
            }

            if (result)
                return true;
        }

        return false;
    }

    Jump::Jump(int offset) : offset(offset)
    {

        if (offset < 1)
            throw TypeException("Illegal jump location");
    }

    TokenReference Jump::run(const TokenList &inputs)
    {

        VERIFY(inputs.size() == 1, "Incorrect number of parameters");

        return create<IntegerScalar>(offset);
    }

    TypeIdentifier Jump::type()
    {
        return GetTypeIdentifier<Jump>();
    }

    ConditionalJump::ConditionalJump(DNF condition, int offset) : Jump(offset), condition(condition)
    {

        VERIFY(condition.size() >= 1, "At least one condition must be defined");
    }

    TokenReference ConditionalJump::run(const TokenList &inputs)
    {

        if (condition.compute(inputs))
        {
            return create<IntegerScalar>(0);
        }
        else
        {
            return create<IntegerScalar>(offset);
        }
    }

    TypeIdentifier ConditionalJump::type()
    {
        return GetTypeIdentifier<ConditionalJump>();
    }

    Sequence<TokenReference> ConditionalJump::serialize() { throw PipelineException("Serialization of jumps supported"); }

    Sequence<TokenReference> Jump::serialize() { throw PipelineException("Serialization of jumps supported"); }

    std::vector<int> topological_sort(const std::vector<std::set<int>> &adjency)
    {
        std::vector<int> levels;
        std::vector<int> indegree;
        std::vector<std::set<int>> reverse;

        reverse.resize(adjency.size());

        indegree.resize(adjency.size(), 0);

        for (size_t i = 0; i < adjency.size(); i++)
        {
            for (int n : adjency[i])
                reverse[n].insert(i);
            indegree[i] = adjency[i].size();
        }

        levels.resize(adjency.size(), -1);

        int level = 0;

        while (true)
        {

            // list to store vertices with indegree 0
            std::vector<int> indegree0;
            for (size_t i = 0; i < adjency.size(); i++)
            {
                if (indegree[i] == 0 && levels[i] < 0)
                    indegree0.push_back(i);
            }
            if (indegree0.empty())
                break;

            for (auto i : indegree0)
            {
                levels[i] = level;
                for (const int n : reverse[i])
                {
                    indegree[n] -= 1;
                }
            }

            level++;
        }

        VERIFY(std::find(levels.begin(), levels.end(), -1) == levels.end(), "Cyclic graph not allowed");

        return levels;
    }

    inline bool all(const std::valarray<bool> va)
    {

        for (auto b : va)
        {
            if (!b)
                return false;
        }

        return true;
    }

    int compare(const std::valarray<bool> &va, const std::valarray<bool> &vb)
    {
        // va < vb : -1, va == vb : 0; va > vb : 1

        for (size_t i = 0; i < MIN(va.size(), vb.size()); i++)
        {
            if (!va[i] && vb[i])
                return -1;
            if (va[i] && !vb[i])
                return 1;
        }

        return va.size() == vb.size() ? 0 : (va.size() < vb.size() ? -1 : 1);
    }

    using OperationData = Pipeline::OperationData;
    class BranchSet
    {
        using BitSet = std::valarray<bool>;
        using Minterm = std::pair<BitSet, BitSet>;

    public:
        BranchSet(size_t size) : _size(size)
        {
        }

        void add(std::map<size_t, bool> &conditions)
        {
            BitSet pos(_size);
            BitSet neg(_size);

            for (auto i : conditions)
            {
                if (i.second)
                {
                    pos[i.first] = true;
                }
                else
                {
                    neg[i.first] = true;
                }
            }

            _insert_minterm(pos, neg);
        }

        void clear()
        {
            _minterms.clear();
        }

        std::set<size_t> used()
        {

            std::set<size_t> r;

            for (auto term : _minterms)
            {
                for (size_t i = 0; i < _size; i++)
                {
                    if (term.first[i] || term.second[i])
                    {
                        r.insert(i);
                    }
                }
            }

            return r;
        }

        inline bool operator<(const BranchSet &b)
        {
            return compare(b) == -1;
        }

        inline int compare(const BranchSet &b)
        {
            for (size_t i = 0; i < MIN(_minterms.size(), b._minterms.size()); i++)
            {
                int comp = _compare_minterm(_minterms[i], b._minterms[i]);
                if (comp != 0)
                    return comp;
            }

            return _minterms.size() == b._minterms.size() ? 0 : (_minterms.size() < b._minterms.size() ? -1 : 1);
        }

        std::pair<std::vector<std::vector<bool>>, std::vector<int>> function()
        {
            std::vector<int> variables;
            std::vector<std::vector<bool>> structure;

            for (auto term : _minterms)
            {
                std::vector<bool> and_terms;
                for (size_t j = 0; j < _size; j++)
                {
                    if (term.first[j])
                    {
                        and_terms.push_back(false);
                        variables.push_back(j);
                    }
                    else if (term.second[j])
                    {
                        and_terms.push_back(true);
                        variables.push_back(j);
                    }
                }
                if (!and_terms.empty())
                    structure.push_back(and_terms);
            }

            return {structure, variables};
        }

        void debug()
        {
            bool first_and = true;
            for (auto term : _minterms)
            {
                std::cout << "(";
                bool first_or = true;
                for (size_t j = 0; j < _size; j++)
                {
                    if (term.first[j])
                    {
                        std::cout << j;
                    }
                    else if (term.second[j])
                    {
                        std::cout << " NOT " << j;
                    }
                    else
                    {
                        continue;
                    }
                    if (first_or)
                    {
                        first_or = false;
                    }
                    else
                    {
                        std::cout << " OR ";
                    }
                }

                std::cout << ")";
                if (first_and)
                {
                    first_and = false;
                }
                else
                {
                    std::cout << " AND ";
                }
            }

            std::cout << std::endl;
        }

    private:
        void _insert_minterm(BitSet &pos, BitSet &neg)
        {

            for (auto i = _minterms.begin(); i != _minterms.end(); i++)
            {
                auto tpos = i->first;
                auto tneg = i->second;

                auto used1 = tpos | tneg;    // Terms used in clause 1
                auto used2 = pos | neg;      // Terms used in clause 2
                auto common = used1 & used2; // All terms used in both clauses

                if (all(used1 == common) && all(used2 == common)) // Same variables used
                {
                    if (all(tpos == pos)) // Same clause
                        return;
                    auto change = tpos ^ pos;
                    if (change.sum() == 1) // We can remove a single independent variable
                    {
                        // del minterms[i];
                        _minterms.erase(i);
                        BitSet npos = tpos & !change;
                        BitSet nneg = tneg & !change;
                        _insert_minterm(npos, nneg);
                        return;
                    }
                }
                else if (all(tpos == (pos & tpos)) && all(tneg == (neg & tneg)))
                {
                    return; // Reduce to clause 1, already merged
                }
                else if (all(pos == (pos & tpos)) && all(neg == (neg & tneg)))
                {
                    _minterms.erase(i);
                    _insert_minterm(pos, neg); // Reduce to clause 2
                    return;
                }
                // Clause not the same, move to next one
            }

            // Not merged, add to list
            _minterms.push_back({pos, neg});
            std::sort(_minterms.begin(), _minterms.end(), _compare_minterm);
        }

        static int _compare_minterm(const Minterm &a, const Minterm &b)
        {
            int c1 = pixelpipes::compare(a.first, b.first);
            int c2 = pixelpipes::compare(a.second, b.second);

            return (c1 == 0) ? c2 : c1;
        }

        size_t _size;
        std::vector<Minterm> _minterms;
    };

    std::set<int> subtree(std::vector<OperationData> &nodes, int root)
    {

        std::set<int> result;
        result.insert(root);

        std::vector<int> current;
        current.push_back(root);

        while (!current.empty())
        {

            std::vector<int> next;

            for (auto i : current)
            {

                for (auto j : nodes[i].inputs)
                {

                    if (result.find(j) != result.end())
                    {
                        result.insert(j);
                        next.push_back(j);
                    }
                }
            }

            current = next;
        }

        return result;
    }

    inline std::set<int> set_intersection(std::set<int> &a, std::set<int> &b)
    {

        std::set<int> result;

        std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::inserter(result, result.begin()));

        return result;
    }

    inline std::set<int> set_union(std::set<int> &a, std::set<int> &b)
    {

        std::set<int> result;

        std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::inserter(result, result.begin()));

        return result;
    }

    inline std::set<int> set_difference(std::set<int> &a, std::set<int> &b)
    {

        std::set<int> result;

        std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::inserter(result, result.begin()));

        return result;
    }

    void build_branches(std::vector<OperationData> &nodes, int current, std::vector<BranchSet> &branches, std::map<int, int> &mapping, std::map<size_t, bool> state = std::map<size_t, bool>())
    {

        if (is_conditional(nodes[current].operation))
        {
            int condition = nodes[current].inputs[2];
            int itrue = nodes[current].inputs[0];
            int ifalse = nodes[current].inputs[1];

            int cposition = mapping[condition];

            bool addtrue = true;
            bool addfalse = false;

            if (state.find(cposition) != state.end())
            {

                if (state[cposition])
                {
                    build_branches(nodes, itrue, branches, mapping, state);
                    addtrue = false;
                }
                else
                {
                    build_branches(nodes, ifalse, branches, mapping, state);
                    addfalse = false;
                }
            }
            else
            {
                state[cposition] = true;
                build_branches(nodes, itrue, branches, mapping, state);
                addtrue = false;

                state[cposition] = false;
                build_branches(nodes, ifalse, branches, mapping, state);
                addfalse = false;
            }

            state.erase(cposition);
            build_branches(nodes, condition, branches, mapping, state);

            if (addtrue)
                build_branches(nodes, itrue, branches, mapping, state);
            if (addfalse)
                build_branches(nodes, ifalse, branches, mapping, state);
        }
        else
        {
            branches[current].add(state);

            for (auto j : nodes[current].inputs)
            {
                build_branches(nodes, j, branches, mapping, state);
            }
        }
    }

    void ensure_order(const std::vector<size_t>& operations, std::vector<std::set<int>>& dependencies)
    {
        for (size_t i = 0; i < operations.size(); i++)
        {
            for (size_t j = 0; j < i; j++)
                dependencies[operations[i]].insert(operations[j]);
        }
    }

    std::vector<OperationData> optimize_pipeline(std::vector<OperationData> &operations, bool predictive, bool prune)
    {

        UNUSED(prune);

        std::vector<size_t> outputs;
        std::vector<size_t> context;
        std::set<int> conditions;
        std::vector<BranchSet> branches;
        std::vector<std::set<int>> dependencies;
        std::vector<TokenReference> stateless;

        dependencies.resize(operations.size());
        stateless.resize(operations.size());

        DEBUGMSG("Optimization start, using %ld operations.\n", operations.size());

        for (size_t i = 0; i < operations.size(); i++)
        {
            // Determine the statelss operation output
            // Note: this at the moment assumes that the operations are already topologically sorted
            
            std::vector<TokenReference> _inputs;

            // Collect special operations: outputs, context, conditionals
            if (is_output(operations[i].operation))
            {
                outputs.push_back(i);
            }
            else if (is_context(operations[i].operation))
            {
                context.push_back(i);
            }
            else if (is_conditional(operations[i].operation))
            {
                conditions.insert(operations[i].inputs[2]);
            }
            for (auto j : operations[i].inputs)
            {
                dependencies[i].insert(j);
            }

            for (int j : operations[i].inputs)
            {
                _inputs.push_back(stateless[j].reborrow());
            }
    
            stateless[i] = operations[i].operation->evaluate(make_view(_inputs));
        }

        branches.resize(operations.size(), BranchSet(conditions.size()));

        std::map<int, int> condition_map;
        std::vector<int> condition_rev;

        size_t p = 0;
        for (auto i : conditions)
        {
            condition_map[i] = p++;
            condition_rev.push_back(i);
        }

        DEBUGMSG("Found %ld outputs.\n", outputs.size());

        for (auto output_node : outputs)
        {
            for (auto i : subtree(operations, output_node))
            {

                if (is_conditional(operations[i].operation))
                {
                    VERIFY(operations[i].inputs.size() == 3, "Illegal conditional operation");

                    int condition = operations[i].inputs[2];
                    int itrue = operations[i].inputs[0];
                    int ifalse = operations[i].inputs[1];

                    // List of nodes required by branch true
                    auto tree_true = subtree(operations, itrue);
                    // List of nodes required by branch false
                    auto tree_false = subtree(operations, ifalse);
                    // List of nodes required to process condition
                    auto tree_condition = subtree(operations, condition);
                    // Required by both branches (A - B) + C
                    auto tree_union = set_union(tree_false, tree_condition);
                    auto common = set_intersection(tree_true, tree_union);

                    if (common.find(condition) == common.end())
                    {
                        auto diff_true = set_difference(tree_true, common);
                        auto diff_false = set_difference(tree_false, common);

                        for (auto j : diff_true)
                        {
                            dependencies[j].insert(condition);
                        }

                        for (auto j : diff_false)
                        {
                            dependencies[j].insert(condition);
                        }
                    }
                }
            }

            if (predictive)
                build_branches(operations, output_node, branches, condition_map);
        }

        // Ensure correct output and context query ordering
        ensure_order(outputs, dependencies);
        ensure_order(context, dependencies);

        /* There are cases where a node without a direct dependency is considered
           redundant in certain branch, we have to add these condition nodes to its
           dependencies to maintain a valid order
        */
        std::vector<size_t> order(operations.size());
        for (size_t i = 0; i < operations.size(); i++)
        {
            if (predictive) {
                for (auto d : branches[i].used())
                {
                    dependencies[i].insert(condition_rev[d]);
                }
                // Remove self-references
                dependencies[i].erase(i);
            }

            // Prepare list for ordering
            order[i] = i;

            // Constants do not have to be conditionally skipped
            if (is_constant(operations[i].operation) || is_context(operations[i].operation))
            {
                dependencies[i].clear();
                branches[i].clear();
            }
        }

        // Order operations according to dependencies and group by branches
        DEBUGMSG("Sorting operations.\n");
        auto levels = topological_sort(dependencies);

        auto comparator = [&](const size_t &a, const size_t &b) -> bool { // a > b
            return std::tie(levels[a], branches[a], a) < std::tie(levels[b], branches[b], b);
        };

        std::sort(order.begin(), order.end(), comparator);

        // Build the pipeline with conditional jumps
        int pending_position = -1;
        std::vector<int> pending_inputs;
        std::vector<std::vector<bool>> pending_clause;
        BranchSet state(conditions.size());

        std::vector<OperationData> optimized;
        std::vector<int> reverse;
        reverse.resize(order.size(), -1);

        DEBUGMSG("Rebuilding pipeline.\n");

        for (auto i : order)
        {
            if (state.compare(branches[i]) != 0)
            {
                if (pending_position >= 0)
                {
                    size_t current_position = optimized.size();
                    int offset = current_position - 1 - pending_position;
                    std::vector<int> remapped_inputs;
                    for (auto d : pending_inputs)
                    {
                        auto m = reverse[condition_rev[d]];
                        if (m < 0)
                        {
                            m = -2;
                        }
                        VERIFY(m >= 0, "Illegal reference");
                        remapped_inputs.push_back(m);
                    }
                    optimized[pending_position] = {create<ConditionalJump>(DNF(pending_clause), offset), remapped_inputs, Metadata()};
                    pending_position = -1;
                }
                auto clauses = branches[i].function();
                if (!clauses.second.empty())
                {
                    optimized.push_back({OperationReference(), Sequence<int>(), Metadata()});
                    pending_position = optimized.size() - 1;
                    pending_clause = clauses.first;
                    pending_inputs = clauses.second;
                }

                state = branches[i];
            }

            std::vector<int> remapped_inputs;
            for (auto d : operations[i].inputs)
            {
                auto m = reverse[d];
                VERIFY(m >= 0, "Illegal reference");
                remapped_inputs.push_back(m);
            }

            optimized.push_back({operations[i].operation.reborrow(), remapped_inputs, operations[i].metadata});
            reverse[i] = optimized.size() - 1;
        }

        return optimized;
    }

}
