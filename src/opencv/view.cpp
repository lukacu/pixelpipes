#include <cmath>
#include <limits>

#include "common.hpp"

namespace pixelpipes
{

    cv::Matx33f translate_view2d(float x, float y)
    {

        cv::Matx33f m(1, 0, x,
                      0, 1, y,
                      0, 0, 1);

        return m;
    }

    PIXELPIPES_OPERATION_AUTO("translate_view2d", translate_view2d);

    cv::Matx33f rotate_view2d(float r)
    {

        cv::Matx33f m(std::cos(r), -std::sin(r), 0,
                      std::sin(r), std::cos(r), 0,
                      0, 0, 1);

        return m;
    }

    PIXELPIPES_OPERATION_AUTO("rotate_view2d", rotate_view2d);

    cv::Matx33f scale_view2d(float x, float y)
    {

        cv::Matx33f m(x, 0, 0,
                      0, y, 0,
                      0, 0, 1);

        return m;
    }

    PIXELPIPES_OPERATION_AUTO("scale_view2d", scale_view2d);

    cv::Matx33f identity_view2d()
    {

        cv::Matx33f m(1, 0, 0,
                      0, 1, 0,
                      0, 0, 1);

        return m;
    }

    PIXELPIPES_OPERATION_AUTO("identity_view2d", identity_view2d);

    TokenReference chain_view2d(const TokenList &inputs)
    {

        VERIFY(inputs.size() > 0, "Incorrect number of parameters");

        cv::Matx33f m(1, 0, 0,
                      0, 1, 0,
                      0, 0, 1);

        for (auto input = inputs.begin(); input != inputs.end(); input++)
        {
            cv::Matx33f view = extract<cv::Matx33f>(*input);

            m = m * view;
        }

        return wrap(m);
    }

    PIXELPIPES_OPERATION("chain_view2d", chain_view2d);

    std::vector<cv::Point2f> view_points2d(const std::vector<cv::Point2f> &points, const cv::Matx33f &transform)
    {
        std::vector<cv::Point2f> points2;
        CV_EX_WRAP(cv::perspectiveTransform(points, points2, transform));
        return points2;
    }

    PIXELPIPES_OPERATION_AUTO("view_points2d", view_points2d);

    /**
     * @brief Returns a view that centers to a bounding box.
     *
     */
    cv::Matx33f center_view2d(const Rectangle &region)
    {
        cv::Matx33f m(1, 0, -(region.left + region.right) / 2,
                      0, 1, -(region.top + region.bottom) / 2,
                      0, 0, 1);

        return m;
    }

    PIXELPIPES_OPERATION_AUTO("center_view2d", center_view2d);

    /**
     * @brief Returns a view that scales space in a way that to a given bounding box
     *
     */
    cv::Matx33f focus_view2d(const Rectangle &region, float scale)
    {
        scale = sqrt(scale / ((region.bottom - region.top) * (region.right - region.left)));

        cv::Matx33f m(scale, 0, 0,
                      0, scale, 0,
                      0, 0, 1);

        return m;
    }

    PIXELPIPES_OPERATION_AUTO("focus_view2d", focus_view2d);

}
