


#include <memory>

#include <pixelpipes/tensor.hpp>

namespace pixelpipes {

    void Tensor::describe(std::ostream &os) const {
        os << "[Tensor]";
    }

    TensorList::TensorList(const View<TensorReference>& inputs) : _data(inputs)
    {
        if (_data.size() > 0) {
            Shape _s = _data[0]->shape();

            for (size_t i = 1; i < _data.size(); i++) {
                _s = _s & _data[i]->shape();
            }

            _shape = _s.push(_data.size());

        } else {
            _shape = Shape(AnyType, make_view(std::vector<Size>{0, unknown, unknown, unknown}));
        }

    }

    TensorList::~TensorList() = default;
    TensorList::TensorList(TensorList &&) = default;
    TensorList &TensorList::operator=(TensorList &&) = default;

    size_t TensorList::length() const
    {
        return _data.size();
    }

    Shape TensorList::shape() const
    {
        return _shape;
    }

    TokenReference TensorList::get(size_t index) const
    {
        return _data[index];
    }

    template <>
    Sequence<TensorReference> extract(const TokenReference& v)
    {
        VERIFY((bool)v, "Uninitialized variable");

        if (v->is<TensorList>()) {
            return (extract<TensorListReference>(v))->elements<TensorReference>();
        } else if (v->is<TensorReference>()) {
            return Sequence<TensorReference>({extract<TensorReference>(v)});
        }

        throw TypeException("Unable to convert to list of images");
    }

/*
    PIXELPIPES_REGISTER_READER(ImageType, [](std::istream &source) -> TokenReference
                               { return read_image(source); });

    PIXELPIPES_REGISTER_READER(ImageListType,
                               [](std::istream &source) -> TokenReference
                               {
                                   try
                                   {
                                       size_t len = read_t<size_t>(source);
                                       std::vector<Image> list;
                                       for (size_t i = 0; i < len; i++)
                                       {
                                           list.push_back(read_image(source));
                                       }
                                       return std::make_shared<ImageList>(make_span(list));
                                   }
                                   catch (std::bad_alloc const &)
                                   {
                                       throw SerializationException("Unable to allocate an array");
                                   }
                               }

    );*/

    /*PIXELPIPES_REGISTER_WRITER(ImageListType, [](TokenReference v, std::ostream &target)
                               {
        auto list = extract<std::vector<Image>>(v);
        write_t(target, list.size());
        for (size_t i = 0; i < list.size(); i++)
        {
            write_image(list[i], target);
        } });*/

    class ConstantTensor : public Operation
    {
    public:
        ConstantTensor(const TensorReference& t)
        {
            tensor = t.reborrow();
        }

        ~ConstantTensor() = default;

        virtual TokenReference run(const TokenList& inputs)
        {
            VERIFY(inputs.size() == 0, "Incorrect number of parameters");
            return tensor;
        }

        virtual Sequence<TokenReference> serialize() { return Sequence<TokenReference>({tensor}); }

    protected:
        TensorReference tensor;
    };

    TokenReference ConstantTensor(TokenList inputs, const TensorReference& image)
    {
        VERIFY(inputs.size() == 0, "Incorrect number of parameters");
        return image;
    }

    // REGISTER_OPERATION_FUNCTION("image", ConstantImage, Image); TODO: support aliases
    //REGISTER_OPERATION_FUNCTION("image_constant", ConstantImage, Image);

    class ConstantImages : public Operation
    {
    public:
        ConstantImages(const Sequence<TensorReference>& images)
        {
            list = create<TensorList>(images);
        }

        ~ConstantImages() = default;

        virtual TokenReference run(const TokenList& inputs)
        {
            VERIFY(inputs.size() == 0, "Incorrect number of parameters");
            return list;
        }

        virtual Sequence<TokenReference> serialize() { return Sequence<TokenReference>({list}); }

    protected:
        Pointer<TensorList> list;
    };

    PIXELPIPES_OPERATION_CLASS("tensor_list", ConstantImages, Sequence<TensorReference>);

}