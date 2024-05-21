
#include <memory>
#include <algorithm>
#include <limits>

#include <xtensor/xtensor.hpp>
#include <xtensor/xtensor_simd.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xnoalias.hpp>

#include <pixelpipes/tensor.hpp>
#include <pixelpipes/operation.hpp>

namespace pixelpipes
{

    PIXELPIPES_REGISTER_ENUM("datatype", DataType);

    struct TensorGuard
    {
        TensorReference guard;
    };

    TensorView::TensorView(const TensorReference &source, const Sizes &shape) : TensorView(source, 0, shape, generate_strides(shape, source->cell_size()))
    {
        VERIFY(size() == source->size(), "Tensor element count mismatch");
    }

    TensorView::TensorView(const TensorReference &source, size_t offset, const Sizes &shape, const Sizes &strides)
    {
        // TODO: can we verify shape somehow?
        // TODO: what if tensor is already a view?
        // VERIFY(!source->is<TensorView>(), "Unable to view existing views");

        VERIFY(shape.size() == strides.size(), "Size mismatch");

        _shape = SizeSequence(shape);
        _strides = SizeSequence(strides);

        _cell_size = source->cell_size();
        _cell_type = source->datatype();

        _size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()) * _cell_size;

        _data = ByteSpan(source->data().data() + offset, source->data().size() - offset);

        _owner = new TensorGuard{source.reborrow()};
        _cleanup = ([](void *v) -> void
                    { delete static_cast<TensorGuard *>(v); });
    }

    TensorView::~TensorView()
    {
        if (_owner)
        {
            _cleanup(_owner);
        }
    }

    Shape TensorView::shape() const
    {
        return Shape(datatype(), _shape);
    }

    size_t TensorView::length() const
    {
        return _shape[0];
    }

    size_t TensorView::cell_size() const
    {
        return _cell_size;
    }

    Type TensorView::datatype() const
    {
        return _cell_type;
    }

    void TensorView::describe(std::ostream &os) const
    {
        os << "[Tensor view]";
    }

    size_t TensorView::size() const
    {
        return _size;
    }

    TokenReference TensorView::get(const Sizes &index) const
    {
        size_t o = get_offset(index);
        return get_scalar(o);
    }

    TokenReference TensorView::get(size_t i) const
    {
        if (_shape.size() == 1)
        {
            return get_scalar(i * cell_size());
        }
        else
        {

            std::vector<size_t> index(_shape.size(), 0);
            index[0] = i;
            size_t offset = get_offset(make_span(index));

            auto ref = pixelpipes::cast<Tensor>(reference());

            return create<TensorView>(ref, offset, make_view(_shape, 1), make_view(_strides, 1));
        }
        return empty();
    }

    ReadonlySliceIterator TensorView::read_slices() const
    {
        return ReadonlySliceIterator(const_data(), _shape, _strides, cell_size());
    }

    WriteableSliceIterator TensorView::write_slices()
    {
        return WriteableSliceIterator(data(), _shape, _strides, cell_size());
    }

    ByteView TensorView::const_data() const
    {
        return ByteView(_data.data(), _data.size());
    }

    SizeSequence TensorView::strides() const
    {
        return _strides;
    }

    ByteSpan TensorView::data()
    {
        return ByteSpan(_data.data(), _data.size());
    }

    template <>
    Sequence<TensorReference> extract(const TokenReference &v)
    {
        VERIFY((bool)v, "Uninitialized variable");

        if (v->is<List>())
        {
            return (extract<ListReference>(v))->elements<TensorReference>();
        }
        else if (v->is<TensorReference>())
        {
            return Sequence<TensorReference>({extract<TensorReference>(v)});
        }

        throw TypeException("Unable to convert to list of images");
    }

    class Stack : public Operation
    {
    public:
        Stack() {}

        virtual TokenReference run(const TokenList &inputs)
        {
            VERIFY(inputs.size() > 1, "Two or more tensors expected");

            TensorReference t0 = extract<TensorReference>(inputs[0]);

            Shape s = t0->shape();

            for (size_t i = 1; i < inputs.size(); i++)
            {
                TensorReference ti = extract<TensorReference>(inputs[i]);

                VERIFY(s == ti->shape(), "Shape mismatch");
            }

            s = s.push(inputs.size());

            TensorReference result = create_tensor(s);

            for (size_t i = 0; i < inputs.size(); i++)
            {
                TensorReference ts = extract<TensorReference>(inputs[i]);
                TensorReference td = extract<TensorReference>(result->get(i));

                copy_buffer(ts, td);
            }

            return result;
        }

        virtual Type type()
        {
            return GetType<Stack>();
        }

        virtual Sequence<TokenReference> serialize() { return Sequence<TokenReference>(); }
    };

    PIXELPIPES_OPERATION_CLASS("stack", Stack);

    /**
     * @brief Converts depth of an image, scaling pixel values.
     */
    TokenReference convert_run(const TensorReference &input, DataType dtype) noexcept(false)
    {
        Type rtype = AnyType;

        switch (dtype)
        {
        case DataType::Boolean:
            rtype = BooleanType;
            break;
        case DataType::Char:
            rtype = CharType;
            break;
        case DataType::Short:
            rtype = ShortType;
            break;
        case DataType::UnsignedShort:
            rtype = UnsignedShortType;
            break;
        case DataType::Integer:
            rtype = IntegerType;
            break;
        case DataType::Float:
            rtype = FloatType;
            break;
        }

        TensorReference output = create_tensor(input->shape().cast(rtype));

        copy_tensor(input, output);

        return output;
    }

    TokenReference convert_eval(const TokenList &inputs)
    {
        VERIFY(inputs.size() == 2, "Two tokens expected");

        auto shape = inputs[0]->shape();

        if (_IS_PLACEHOLDER(inputs[1])) {
            return create<Placeholder>(shape.cast(AnyType));
        }

        DataType dtype = extract<DataType>(inputs[1]);

        switch (dtype)
        {
        case DataType::Boolean:
            return create<Placeholder>(shape.cast(BooleanType));
        case DataType::Char:
            return create<Placeholder>(shape.cast(CharType));
        case DataType::Short:
            return create<Placeholder>(shape.cast(ShortType));
        case DataType::UnsignedShort:
            return create<Placeholder>(shape.cast(UnsignedShortType));
        case DataType::Integer:
            return create<Placeholder>(shape.cast(IntegerType));
        case DataType::Float:         
            return create<Placeholder>(shape.cast(FloatType));  
        }

        throw TypeException("Unsupported data type");
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("convert", convert_run, convert_eval);

    /**
     * @brief Flip tensor along specified axis.
     */
/*     
    TensorReference flip_run(const TensorReference &input, int axis) noexcept(false)
    {
        Sizes shape = input->shape().sizes();
        Sizes strides = input->strides();

        Sizes new_strides = strides;

        std::reverse(new_strides.begin(), new_strides.end());

        TensorReference output = create<TensorView>(input, 0, shape, new_strides);


        return output;
    }

    TokenReference flip_eval(const TokenList &inputs)
    {
        VERIFY(inputs.size() == 2, "Two tokens expected");

        auto shape = inputs[0]->shape();

        if (shape.is_anything())
            return create<Placeholder>(shape);

        if (!_IS_PLACEHOLDER(inputs[1])) {
            int axis = extract<int>(inputs[1]);
            VERIFY(axis >= 0 && axis < shape.rank(), "Axis out of range");
        }

        return create<Placeholder>(shape);
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("flip", flip_run, flip_eval);
*/
    /**
     * @brief Reshape tensor.
     */

    TokenReference reshape_run(const TensorReference &input, const Sequence<int> shape) noexcept(false)
    {
        Shape s = input->shape();

        // Check if the number of elements is the same by multiplying all the dimensions
        size_t n = 1;
        SizeSequence new_shape(shape.size());
        for (size_t i = 0; i < shape.size(); i++)
        {
            VERIFY(shape[i] > 0, "Invalid shape");
            n *= shape[i];
            new_shape[i] = shape[i];
        }

        VERIFY(n == s.size(), "Number of elements must remain the same");

        if (input->contiguous())
        {
            return create<TensorView>(input, new_shape);
        }
        else 
        {
            auto output = create_tensor(Shape(input->datatype(), new_shape));
            copy_tensor(input, output);

            return output.reborrow();
        }
    }

    TokenReference reshape_eval(const TokenList &inputs)
    {
        VERIFY(inputs.size() >= 2, "At least two tokens expected");

        auto s = inputs[0]->shape();

        if (s.is_anything() || _IS_PLACEHOLDER(inputs[1])) {
            return create<Placeholder>();
        }

        auto shape = extract<Sequence<int>>(inputs[1]);

        size_t n = 1;
        SizeSequence new_shape(shape.size());
        for (size_t i = 0; i < shape.size(); i++)
        {
            n *= shape[i];
            new_shape[i] = shape[i];
        }

        VERIFY(n == s.size(), "Number of elements must remain the same");

        return create<Placeholder>(Shape(s.element(), new_shape));
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("reshape", reshape_run, reshape_eval);

    /**
     * @brief Transpose tensor.
     */
    
    inline bool is_permutation(const Sequence<int> &v)
    {
        std::vector<int> permutation(v.size(), 0);
        for (size_t i = 0; i < v.size(); i++)
        {
            permutation[i] = i;
        }
        for (size_t i = 0; i < v.size(); i++)
        {
            if (std::find(v.begin(), v.end(), i) == v.end())
                return false;
        }
        return true;
    }

    TokenReference transpose_run(const TensorReference &input, const Sequence<int> axes) noexcept(false)
    {
        Shape shape = input->shape();
        SizeSequence strides = input->strides();

        VERIFY(axes.size() == shape.rank(), "Axes must have the same rank as the tensor");
        VERIFY(is_permutation(axes), "Axes must be a permutation of [0, 1, ..., rank-1]");

        SizeSequence new_shape(shape.rank());
        SizeSequence new_strides(shape.rank());

        for (size_t i = 0; i < shape.rank(); i++)
        {
            new_shape[i] = shape[axes[i]];
            new_strides[i] = strides[axes[i]];
        }

        return create<TensorView>(input, 0, new_shape, new_strides);
    }

    TokenReference transpose_eval(const TokenList &inputs)
    {
        VERIFY(inputs.size() == 1, "One token expected");

        auto shape = inputs[0]->shape();

        if (shape.is_anything())
            return create<Placeholder>(shape);

        if (_IS_PLACEHOLDER(inputs[1])) {
            return create<Placeholder>();
        }

        auto axes = extract<Sequence<int>>(inputs[1]);

        // Verify that axes is a permutation of [0, 1, ..., rank-1]
        VERIFY(is_permutation(axes), "Axes must be a permutation of [0, 1, ..., rank-1]");

        SizeSequence new_shape(shape.rank());
 
        for (size_t i = 0; i < shape.rank(); i++)
        {
            new_shape[i] = shape[axes[i]];
        }

        return create<Placeholder>(shape.element(), new_shape);
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("transpose", transpose_run, transpose_eval);

}
