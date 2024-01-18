#pragma once

#include <vector>
#include <algorithm>

#include <pixelpipes/pipeline.hpp>

namespace pixelpipes
{

    class DNF
    {
    public:
        DNF();
        DNF(std::vector<std::vector<bool>> clauses);

        ~DNF();

        size_t size() const;

        bool compute(TokenList values) const;

    private:
        std::vector<std::vector<bool>> _clauses;

        size_t _size;
    };

    class Jump : public Operation
    {
    public:
        Jump(int offset);

        ~Jump() = default;

        virtual TokenReference run(const TokenList &inputs);

        virtual TypeIdentifier type();

        virtual Sequence<TokenReference> serialize();

    protected:
        int offset;
    };

    class ConditionalJump : public Jump
    {
    public:
        ConditionalJump(DNF condition, int offset);

        ~ConditionalJump() = default;

        virtual TokenReference run(const TokenList &inputs);

        virtual TypeIdentifier type();

        virtual Sequence<TokenReference> serialize();

    private:
        DNF condition;
    };

    /**
     * The main function that optimizes a pipeline and returns a new set of operations. The optimization can be 
     * just ordering, it can add predictive jumps and preexecute certain (deterministic) parts of pipeline and compact them
     * to constants.
    */
    std::vector<Pipeline::OperationData> optimize_pipeline(std::vector<Pipeline::OperationData> &operations, bool predictive = true, bool prune = true);

}