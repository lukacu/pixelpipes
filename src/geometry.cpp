#include <pixelpipes/operation.hpp>
#include <pixelpipes/geometry.hpp>

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

    PIXELPIPES_OPERATION_AUTO("bounding_box", bounding_box);

    /**
     * @brief Returns a bounding box of custom size, defined by four inputs.
     *
     */
    Rectangle make_rectangle(float x1, float y1, float x2, float y2)
    {
        Rectangle rectangle{(float)x1, (float)y1, (float)x2, (float)y2};
        return rectangle;
    }

    PIXELPIPES_OPERATION_AUTO("make_rectangle", make_rectangle);

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

    PIXELPIPES_OPERATION_AUTO("points2d_center", points2d_center);

    std::vector<Point2D> points_from_rectangle(Rectangle r)
    {
        return std::vector<Point2D>({Point2D{r.left, r.top}, Point2D{r.right, r.top}, Point2D{r.right, r.bottom}, Point2D{r.left, r.bottom}});
    }

    PIXELPIPES_OPERATION_AUTO("points_from_rectangle", points_from_rectangle);

    Point2D make_point2d(float x, float y)
    {
        return Point2D{x, y};
    }

    PIXELPIPES_OPERATION_AUTO("make_point2d", make_point2d);

    TokenReference make_points2d(const TokenList &inputs)
    {

        VERIFY(inputs.size() % 2 == 0, "Incorrect number of parameters, number should be even");

        std::vector<Point2D> result;

        for (size_t i = 0; i < inputs.size(); i += 2)
        {
            float x = extract<float>(inputs[i]);
            float y = extract<float>(inputs[i + 1]);
            result.push_back(Point2D{x, y});
        }

        return wrap(result);
    }

    PIXELPIPES_OPERATION("make_points2d", make_points2d);
    /*
    TokenReference PointsFromList(TokenList inputs) {

        VERIFY(inputs.size() == 1, "Incorrect number of parameters");
        std::vector<Point2D> result;

        ListReference list = extract<ListReference>(inputs[0]);

        VERIFY(is_numeric_list(list) && list->length() % 2 == 0, "Not a float list with even element count");

        for (size_t i = 0; i < list->length(); i+=2) {
            float x = extract<float>(list->get(i));
            float y = extract<float>(list->get(i+1));
            result.push_back(Point2D{x, y});
        }

        return std::make_shared<Point2DList>(make_span(result));

    }

    PIXELPIPES_OPERATION_AUTO("list_to_points", PointsFromList);
    */
    std::vector<Point2D> random_points2d(int count, int seed)
    {

        std::vector<Point2D> data(count);

        RandomGenerator generator = create_generator(seed);

        std::uniform_real_distribution<float> distribution(-1, 1);

        for (int i = 0; i < count; i++)
        {
            data.push_back(Point2D{distribution(generator), distribution(generator)});
        }

        return data;
    }

    PIXELPIPES_OPERATION_AUTO("random_points2d", random_points2d);

    Point2D point2d_add(const Point2D &point0, const Point2D &point1)
    {
        Point2D result;
        result.x = point0.x + point1.x;
        result.y = point0.y + point1.y;
        return result;
    }

    PIXELPIPES_OPERATION_AUTO("point2d_add", point2d_add);

    Point2D point2d_subtract(const Point2D &point0, const Point2D &point1)
    {
        Point2D result;
        result.x = point0.x - point1.x;
        result.y = point0.y - point1.y;
        return result;
    }

    PIXELPIPES_OPERATION_AUTO("point2d_subtract", point2d_subtract);

    Point2D point2d_multiply(const Point2D &point0, const Point2D &point1)
    {
        Point2D result;
        result.x = point0.x * point1.x;
        result.y = point0.y * point1.y;
        return result;
    }

    PIXELPIPES_OPERATION_AUTO("point2d_multiply", point2d_multiply);

    Point2D point2d_divide(const Point2D &point0, const Point2D &point1)
    {
        Point2D result;
        result.x = point0.x / point1.x;
        result.y = point0.y / point1.y;
        return result;
    }

    PIXELPIPES_OPERATION_AUTO("point2d_divide", point2d_divide);

    Point2D point2d_power(const Point2D &point0, const Point2D &point1)
    {
        Point2D result;
        result.x = pow(point0.x, point1.x);
        result.y = pow(point0.y, point1.y);
        return result;
    }

    PIXELPIPES_OPERATION_AUTO("point2d_power", point2d_power);

    template <Point2D (*F)(const Point2D &, const Point2D &)>
    Sequence<Point2D> points2d_elementwise_binary(const Sequence<Point2D> &points0, const Sequence<Point2D> &points1)
    {
        if (points0.size() == points1.size())
        {
            Sequence<Point2D> result(points1.size());
            for (size_t i = 0; i < points0.size(); i++)
            {
                result[i] = F(points0[i], points1[i]);
            }
            return result;
        }
        else if (points0.size() == 1)
        {
            Sequence<Point2D> result(points1.size());
            for (size_t i = 0; i < points1.size(); i++)
            {
                result[i] = F(points0[0], points1[i]);
            }
            return result;
        }
        else if (points1.size() == 1)
        {
            Sequence<Point2D> result(points0.size());
            for (size_t i = 0; i < points0.size(); i++)
            {
                result[i] = F(points0[i], points1[0]);
            }
            return result;
        }

        throw IllegalStateException("List sizes do not match");
    }

#define points2d_add points2d_elementwise_binary<point2d_add>
    PIXELPIPES_OPERATION_AUTO("points2d_add", points2d_add);

#define points2d_subtract points2d_elementwise_binary<point2d_subtract>
    PIXELPIPES_OPERATION_AUTO("points2d_subtract", points2d_subtract);

#define points2d_multiply points2d_elementwise_binary<point2d_multiply>
    PIXELPIPES_OPERATION_AUTO("points2d_multiply", points2d_multiply);

#define points2d_divide points2d_elementwise_binary<point2d_divide>
    PIXELPIPES_OPERATION_AUTO("points2d_divide", points2d_divide);

#define points2d_power points2d_elementwise_binary<point2d_power>
    PIXELPIPES_OPERATION_AUTO("points2d_power", points2d_power);

}