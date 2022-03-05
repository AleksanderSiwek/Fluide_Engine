#include <gtest/gtest.h>
#include "../src/common/vector3.hpp"


TEST(Vector3Test, Constructor_test)
{
    Vector3<int> v;
    EXPECT_EQ(0, (v.x && v.y && v.z));
}

TEST(Vector3Test, SingleValConstructor_test)
{
    Vector3<int> v(1);
    EXPECT_EQ(1, v.x);
    EXPECT_EQ(1, v.y);
    EXPECT_EQ(1, v.z);
}

TEST(Vector3Test, ValConstructor_test)
{
    Vector3<int> v(1, 2, 3);
    EXPECT_EQ(1, v.x);
    EXPECT_EQ(2, v.y);
    EXPECT_EQ(3, v.z);
}

TEST(Vector3Test, CopyConstructor_test)
{
    Vector3<int> v(1, 2, 3);
    Vector3<int> v1(v);
    EXPECT_EQ(1, v1.x);
    EXPECT_EQ(2, v1.y);
    EXPECT_EQ(3, v1.z);
}

TEST(Vector3Test, AddVal_test)
{
    Vector3<int> v(1, 2, 3);
    v.Add(int(4));
    EXPECT_EQ(5, v.x);
    EXPECT_EQ(6, v.y);
    EXPECT_EQ(7, v.z);
}

TEST(Vector3Test, AddVect_test)
{
    Vector3<int> v(1, 2, 3);
    Vector3<int> v1(2, 3, 4);
    v.Add(v1);
    EXPECT_EQ(3, v.x);
    EXPECT_EQ(5, v.y);
    EXPECT_EQ(7, v.z);
}

TEST(Vector3Test, SubtractVal_test)
{
    Vector3<int> v(1, 2, 3);
    v.Subtract(int(4));
    EXPECT_EQ(-3, v.x);
    EXPECT_EQ(-2, v.y);
    EXPECT_EQ(-1, v.z);
}

TEST(Vector3Test, SubtractVect_test)
{
    Vector3<int> v(1, 2, 3);
    Vector3<int> v1(2, 3, 4);
    v.Subtract(v1);
    EXPECT_EQ(-1, v.x);
    EXPECT_EQ(-1, v.y);
    EXPECT_EQ(-1, v.z);
}

TEST(Vector3Test, MultiplyVal_test)
{
    Vector3<int> v(1, 2, 3);
    v.Multiply(int(4));
    EXPECT_EQ(4, v.x);
    EXPECT_EQ(8, v.y);
    EXPECT_EQ(12, v.z);
}

TEST(Vector3Test, MultiplyVect_test)
{
    Vector3<int> v(1, 2, 3);
    Vector3<int> v1(2, 3, 4);
    v.Multiply(v1);
    EXPECT_EQ(2, v.x);
    EXPECT_EQ(6, v.y);
    EXPECT_EQ(12, v.z);
}

TEST(Vector3Test, DivideVal_test)
{
    Vector3<double> v(1, 2, 3);
    v.Divide(double(4));
    EXPECT_EQ(0.25, v.x);
    EXPECT_EQ(0.5, v.y);
    EXPECT_EQ(0.75, v.z);
}

TEST(Vector3Test, DivideVect_test)
{
    Vector3<double> v(1, 2, 2);
    Vector3<double> v1(1, 4, 8);
    v.Divide(v1);
    EXPECT_EQ(1, v.x);
    EXPECT_EQ(0.5, v.y);
    EXPECT_EQ(0.25, v.z);
}

TEST(Vector3Test, GetLength_test)
{
    Vector3<double> v(3, 5, sqrt(2));
    EXPECT_EQ(6, v.GetLength());
}

TEST(Vector3Test, Max_test)
{
    Vector3<int> v(1, 2, 3);
    EXPECT_EQ(3, v.Max());
}

TEST(Vector3Test, AbsMax_test)
{
    Vector3<int> v(1, 2, -3);
    EXPECT_EQ(3, v.AbsMax());
}

TEST(Vector3Test, Min_test)
{
    Vector3<int> v(1, 2, 3);
    EXPECT_EQ(1, v.Min());
}

TEST(Vector3Test, Dot_test)
{
    Vector3<double> v(1, 2, 3);
    Vector3<double> v1(1, 2, 3);
    EXPECT_EQ(14, v.Dot(v1));
}

TEST(Vector3Test, Normalize_test)
{
    Vector3<double> v(1, 1, sqrt(2));
    v.Normalize();
    EXPECT_EQ(0.5, v.x);
    EXPECT_EQ(0.5, v.y);
    EXPECT_EQ(double(sqrt(2)/2), v.z);
}

TEST(Vector3Test, GetNormalized_test)
{
    Vector3<double> v(1, 1, sqrt(2));
    Vector3<double> v_norm = v.GetNormalized();

    EXPECT_EQ(0.5, v_norm.x);
    EXPECT_EQ(0.5, v_norm.y);
    EXPECT_EQ(double(sqrt(2)/2), v_norm.z);
    EXPECT_EQ(1, v.x);
    EXPECT_EQ(1, v.y);
    EXPECT_EQ(double(sqrt(2)), v.z);
}

TEST(Vector3Test, IsEqualVect_test)
{
    Vector3<int> v(1, 2, 4);
    Vector3<int> v1(1, 2, 4);
    EXPECT_EQ(true, v.IsEqual(v1));
}

TEST(Vector3Test, IsEqualVal_test)
{
    Vector3<int> v(5, 5, 5);
    EXPECT_EQ(true, v.IsEqual(5));
}

TEST(Vector3Test, AssignmentOperatorVect_test)
{
    Vector3<int> v(1, 2, 4);
    Vector3<int> v1 = v;
    EXPECT_EQ(1, v1.x);
    EXPECT_EQ(2, v1.y);
    EXPECT_EQ(4, v1.z);
}

TEST(Vector3Test, AssignmentOperatorVal_test)
{
    Vector3<int> v = (1, 2, 3);
    v = 5;
    EXPECT_EQ(5, v.x);
    EXPECT_EQ(5, v.y);
    EXPECT_EQ(5, v.z);
}

TEST(Vector3Test, PlusEqualValOperator_test)
{
    Vector3<int> v(1, 2, 4);
    v += 1;
    EXPECT_EQ(2, v.x);
    EXPECT_EQ(3, v.y);
    EXPECT_EQ(5, v.z);
}

TEST(Vector3Test, PlusEqualVectOperator_test)
{
    Vector3<int> v(1, 2, 4);
    Vector3<int> v1(1, 1, 1);
    v1 += v;
    EXPECT_EQ(2, v1.x);
    EXPECT_EQ(3, v1.y);
    EXPECT_EQ(5, v1.z);
}

TEST(Vector3Test, MinusEqualValOperator_test)
{
    Vector3<int> v(1, 2, 4);
    v -= 1;
    EXPECT_EQ(0, v.x);
    EXPECT_EQ(1, v.y);
    EXPECT_EQ(3, v.z);
}

TEST(Vector3Test, MinusEqualVectOperator_test)
{
    Vector3<int> v(1, 2, 4);
    Vector3<int> v1(1, 1, 1);
    v1 -= v;
    EXPECT_EQ(0, v1.x);
    EXPECT_EQ(-1, v1.y);
    EXPECT_EQ(-3, v1.z);
}

TEST(Vector3Test, MultiplyEqualValOperator_test)
{
    Vector3<int> v(1, 2, 4);
    v *= 2;
    EXPECT_EQ(2, v.x);
    EXPECT_EQ(4, v.y);
    EXPECT_EQ(8, v.z);
}

TEST(Vector3Test, MultiplyEqualVectOperator_test)
{
    Vector3<int> v(1, 2, 4);
    Vector3<int> v1(1, 1, 1);
    v1 *= v;
    EXPECT_EQ(1, v1.x);
    EXPECT_EQ(2, v1.y);
    EXPECT_EQ(4, v1.z);
}

TEST(Vector3Test, DivideEqualValOperator_test)
{
    Vector3<double> v(1, 2, 4);
    v /= 4;
    EXPECT_EQ(0.25, v.x);
    EXPECT_EQ(0.5, v.y);
    EXPECT_EQ(1, v.z);
}

TEST(Vector3Test, DivideEqualVectOperator_test)
{
    Vector3<double> v(1, 2, 4);
    Vector3<double> v1(1, 1, 1);
    v1 /= v;
    EXPECT_EQ(1, v1.x);
    EXPECT_EQ(0.5, v1.y);
    EXPECT_EQ(0.25, v1.z);
}

TEST(Vector3Test, EqualEqualOperatorVect_test)
{
    Vector3<double> v(1, 2, 4);
    Vector3<double> v1(1, 2, 3);
    EXPECT_EQ(false, v == v1);
}

TEST(Vector3Test, EqualEqualOperatorVal_test)
{
    Vector3<double> v(1, 2, 4);
    EXPECT_EQ(false, v == 1);
}

TEST(Vector3Test, NotEqualOperatorVect_test)
{
    Vector3<double> v(1, 2, 4);
    Vector3<double> v1(1, 2, 3);
    EXPECT_EQ(true, v != v1);
}

TEST(Vector3Test, NotEqualOperatorVal_test)
{
    Vector3<double> v(1, 2, 4);
    EXPECT_EQ(true, v != 1);
}

TEST(Vector3Test, PlusValOperator_test)
{
    Vector3<int> v(1, 2, 4);
    Vector3<int> v1 = v + 1;
    EXPECT_EQ(2, v1.x);
    EXPECT_EQ(3, v1.y);
    EXPECT_EQ(5, v1.z);
}

TEST(Vector3Test, PlusVectVectOperator_test)
{
    Vector3<int> v(1, 2, 4);
    Vector3<int> v1(1, 1, 1);
    Vector3<int> v2 = v + v1;
    EXPECT_EQ(2, v2.x);
    EXPECT_EQ(3, v2.y);
    EXPECT_EQ(5, v2.z);
}

TEST(Vector3Test, ValMinusVectOperator_test)
{
    Vector3<int> v(1, 2, 4);
    v = 1 - v;
    EXPECT_EQ(0, v.x);
    EXPECT_EQ(-1, v.y);
    EXPECT_EQ(-3, v.z);
}

TEST(Vector3Test, VectMinusValOperator_test)
{
    Vector3<int> v(1, 2, 4);
    v = v - 1;
    EXPECT_EQ(0, v.x);
    EXPECT_EQ(1, v.y);
    EXPECT_EQ(3, v.z);
}

TEST(Vector3Test, VectMinusVectOperator_test)
{
    Vector3<int> v(1, 2, 4);
    Vector3<int> v1(1, 2, 1);
    v = v - v1;
    EXPECT_EQ(0, v.x);
    EXPECT_EQ(0, v.y);
    EXPECT_EQ(3, v.z);
}

TEST(Vector3Test, ValMultiplyVectOperator_test)
{
    Vector3<int> v(1, 2, 4);
    v = 2 * v;
    EXPECT_EQ(2, v.x);
    EXPECT_EQ(4, v.y);
    EXPECT_EQ(8, v.z);
}

TEST(Vector3Test, VectMultiplyValOperator_test)
{
    Vector3<int> v(1, 2, 4);
    v = v * 2;
    EXPECT_EQ(2, v.x);
    EXPECT_EQ(4, v.y);
    EXPECT_EQ(8, v.z);
}

TEST(Vector3Test, VectMultiplyVectOperator_test)
{
    Vector3<int> v(1, 2, 4);
    Vector3<int> v1(1, 2, 1);
    v = v * v1;
    EXPECT_EQ(1, v.x);
    EXPECT_EQ(4, v.y);
    EXPECT_EQ(4, v.z);
}

TEST(Vector3Test, ValDivideVectOperator_test)
{
    Vector3<int> v(2, 4, 8);
    v = 8 / v;
    EXPECT_EQ(4, v.x);
    EXPECT_EQ(2, v.y);
    EXPECT_EQ(1, v.z);
}

TEST(Vector3Test, VectDivideValOperator_test)
{
    Vector3<int> v(2, 2, 4);
    v = v / 2;
    EXPECT_EQ(1, v.x);
    EXPECT_EQ(1, v.y);
    EXPECT_EQ(2, v.z);
}

TEST(Vector3Test, VectDivideVectOperator_test)
{
    Vector3<int> v(1, 2, 4);
    Vector3<int> v1(1, 2, 2);
    v = v / v1;
    EXPECT_EQ(1, v.x);
    EXPECT_EQ(1, v.y);
    EXPECT_EQ(2, v.z);
}

