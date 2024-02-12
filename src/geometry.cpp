
#include <pixelpipes/geometry.hpp>
#include <pixelpipes/operation.hpp>

namespace pixelpipes
{

    /**
     * @brief Calculates bounding box of a list of points and returns it as a list of four numbers: left, top, right, bottom.
     *
     */

    Rectangle bounding_box(const Sequence<Point2D> &points)
    {

        float top = std::numeric_limits<float>::max();
        float bottom = std::numeric_limits<float>::lowest();
        float left = std::numeric_limits<float>::max();
        float right = std::numeric_limits<float>::lowest();

        for (auto p : points)
        {
            top = MIN(top, p.y);
            left = MIN(left, p.x);
            bottom = MAX(bottom, p.y);
            right = MAX(right, p.x);
        }

        return {left, top, right, bottom};
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("bounding_box", bounding_box, (constant_shape<float, 4>));

    /**
     * @brief Returns a bounding box of custom size, defined by four inputs.
     *
     */
    Rectangle make_rectangle(float x1, float y1, float x2, float y2)
    {
        Rectangle rectangle{(float)x1, (float)y1, (float)x2, (float)y2};
        return rectangle;
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("make_rectangle", make_rectangle, (constant_shape<float, 4>));

    Point2D points2d_center(const Sequence<Point2D> &list)
    {

        Point2D accumulator{0, 0};
        for (size_t i = 0; i < list.size(); i++)
        {
            auto p = list[i];
            accumulator.x += p.x;
            accumulator.y += p.y;
        }
        accumulator.x /= (float)list.size();
        accumulator.y /= (float)list.size();

        return accumulator;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("points2d_center", points2d_center, (constant_shape<float, 1, 2>));

    Sequence<Point2D> points_from_rectangle(Rectangle r)
    {
        return Sequence<Point2D>({Point2D{r.left, r.top}, Point2D{r.right, r.top}, Point2D{r.right, r.bottom}, Point2D{r.left, r.bottom}});
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("points_from_rectangle", points_from_rectangle, (constant_shape<float, 4, 2>));

    Point2D make_point2d(float x, float y)
    {
        return Point2D{x, y};
    }

    PIXELPIPES_UNIT_OPERATION_AUTO("make_point2d", make_point2d, (constant_shape<float, 1, 2>));

    TokenReference make_points2d(const TokenList &inputs)
    {

        VERIFY(inputs.size() % 2 == 0, "Incorrect number of parameters, number should be even");

        Sequence<Point2D> result(inputs.size() / 2);

        for (size_t i = 0; i < inputs.size(); i += 2)
        {
            float x = extract<float>(inputs[i]);
            float y = extract<float>(inputs[i + 1]);
            result[i / 2] = (Point2D{x, y});
        }

        return wrap(result);
    }

    PIXELPIPES_COMPUTE_OPERATION("make_points2d", make_points2d, (constant_shape<float, unknown, 2>));
   
    Sequence<Point2D> random_points2d(int count, int seed)
    {

        Sequence<Point2D> data((size_t)count);

        RandomGenerator generator = create_generator(seed);

        std::uniform_real_distribution<float> distribution(-1, 1);

        for (int i = 0; i < count; i++)
        {
            data[i] = (Point2D{distribution(generator), distribution(generator)});
        }

        return data;
    }

    PIXELPIPES_COMPUTE_OPERATION_AUTO("random_points2d", random_points2d, (constant_shape<float, unknown, 2>));

}