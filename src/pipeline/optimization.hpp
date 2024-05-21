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

        virtual Type type();

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

        virtual Type type();

        virtual Sequence<TokenReference> serialize();

    private:
        DNF condition;
    };

    /**
     * The function evaluates the results of individual operations in a stateless manner. This can be used to
     * optimize the pipeline by removing operations that are fixed and can be precomputed. It can also be used
     * to determine shapes of the outputs of the operations.
    */
    std::vector<TokenReference> stateless_pass(std::vector<Pipeline::OperationData> &operations);

    /**
     * The main function that optimizes a pipeline and returns a new set of operations. The optimization can be 
     * just ordering, it can add predictive jumps and merge parts of the graph that are repeated.
    */
    std::vector<Pipeline::OperationData> optimize_pipeline(std::vector<Pipeline::OperationData> &operations, bool predictive = true, bool merge = true);

}