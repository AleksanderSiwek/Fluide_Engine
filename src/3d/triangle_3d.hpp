#ifndef _TRAINGLE_3D_HPP
#define _TRAINGLE_3D_HPP


struct Triangle3D
{
    size_t point1Idx;
    size_t point2Idx;
    size_t point3Idx;
    size_t normalIdx;
    
    Triangle3D(size_t p1=0, size_t p2=1, size_t p3=2, size_t normIdx=0) 
        : point1Idx(p1), point2Idx(p2), point3Idx(p3), normalIdx(normIdx) {}
    Triangle3D(const struct Triangle3D& other) 
        : point1Idx(other.point1Idx), point2Idx(other.point2Idx), point3Idx(other.point3Idx), normalIdx(other.normalIdx) {}
};

typedef struct Triangle3D Triangle3D_t;

#endif // _TRAINGLE_3D_HPP