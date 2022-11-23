//
// Created by gaoxiang on 19-5-4.
//

#ifndef MYSLAM_ALGORITHM_H
#define MYSLAM_ALGORITHM_H

// algorithms used in myslam
#include "myslam/common_include.h"

/*
 //A：4✖4, P:4x1(X,Y,Z,1)
   //在这里解释一下原理
   //point1(归一化像素坐标）= 1/s1*K*[I|0]*pt_world， point1 = (u1, v1)
   //point2(归一化像素坐标）= 1/s2*K*[R21|t]*pt_world, point2=(u2, v2) //相对坐标变换
   //pr1 = K*[I|0], pr2 = K[R21|t], 就是这里的pose
   //下面矩阵中的";"表示换行
   //写成行向量pr1 = [pr1,1; pr1,2; pr1,3], pr2 = [pr2,1; pr2,2; pr2,3]
   //所以上面的变换成了[s1u1; s1v1; s1] = [pr1,1; pr1,2; pr1,3]P1, point2一样
   //把s1 = pr1,3*P1代入前两行，消掉第3行，得到[pr1,3*P1*u1] = [pr1,1*P1]，后面同样
   //把point1, point2的4个等式组合，得到
   //[pr1,3*u1 - pr1,1]
   //[Pr1,3*v1 - pr1,2] * P1 = 0, 左边4x4的矩阵就是函数中的A,pr为函数中的m
   //[pr2,3*u2 - pr2,1]
   //[pr2,3*v2 - pr2,2]
*/

namespace myslam {

/**
 * linear triangulation with SVD
 * @param poses     poses,
 * @param points    points in normalized plane
 * @param pt_world  triangulated point in the world
 * @return true if success
 */
inline bool triangulation(const std::vector<SE3> &poses,
                   const std::vector<Vec3> points, Vec3 &pt_world) {
    MatXX A(2 * poses.size(), 4);
    VecX b(2 * poses.size());
    b.setZero();
    for (size_t i = 0; i < poses.size(); ++i) {
        Mat34 m = poses[i].matrix3x4();   //映射矩阵
        A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
        A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
    }
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    //A(4x4) = UWV^T, 即A[V1, V2, V3, V4] = U*diag(sigma1, sigma2, sigma3, sigma4), sigma4 = 0
    //满足A*V4 = 0, SVD分解得到的V矩阵的最后一列作为P1的解
    pt_world = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

    if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
        // 解质量不好，放弃
        return true;
    }
    return false;
}

// converters
inline Vec2 toVec2(const cv::Point2f p) { return Vec2(p.x, p.y); }

}  // namespace myslam

#endif  // MYSLAM_ALGORITHM_H
