//
// Created by gaoxiang on 19-5-2.
//
#pragma once

#ifndef MYSLAM_FEATURE_H
#define MYSLAM_FEATURE_H

#include <memory>   //定义智能指针的头文件
#include <opencv2/features2d.hpp>
#include "myslam/common_include.h"

namespace myslam {

struct Frame;
struct MapPoint;

/**
 * 2D 特征点
 * 在三角化之后会被关联一个地图点
 */
struct Feature {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Feature> Ptr;

    /*
weak_ptr是标准库对shared_ptr的补充，想象这样一个场景，
两个类互相持有对方类的shared_ptr，则造成两方由于shared_ptr引用计数不为0而导致无法析构，
从而产生内存泄漏。weak_ptr则并不共享资源，只是获取shared_ptr的观测权，
它的构造不会引起指针引用计数的增加，这是通过使用成员函数lock()从被观测的shared_ptr获得一个可用的shared_ptr对象，
 从而操作资源。除此之外，成员函数use_count()可以观测资源的引用计数，
 成员函数expired()为true则表示被观测的资源(也就是shared_ptr的管理的资源)已经销毁。

    */

    std::weak_ptr<Frame> frame_;         // 持有该feature的frame
    cv::KeyPoint position_;              // 2D提取位置
    std::weak_ptr<MapPoint> map_point_;  // 关联地图点

    bool is_outlier_ = false;       // 是否为异常点
    bool is_on_left_image_ = true;  // 标识是否提在左图，false为右图

   public:
    Feature() {}

    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp)
        : frame_(frame), position_(kp) {}
};
}  // namespace myslam

#endif  // MYSLAM_FEATURE_H
