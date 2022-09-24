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

    std::vector<Pipeline::OperationData> optimize_pipeline(std::vector<Pipeline::OperationData> &operations, bool predictive = true);

}