#include <gtest/gtest.h>
#include "../src/common/array3.hpp"


TEST(Array3Test, DefaultConstructor_test)
{
    Array3<int> arr;
    Vector3<size_t> size = arr.GetSize();
    EXPECT_EQ(0, size.x);
    EXPECT_EQ(0, size.y);
    EXPECT_EQ(0, size.z);
}

TEST(Array3Test, ConstructorSizeT_test)
{
    Array3<int> arr(1, 2, 3);
    Vector3<size_t> size = arr.GetSize();
    EXPECT_EQ(1, size.x);
    EXPECT_EQ(2, size.y);
    EXPECT_EQ(3, size.z);
    EXPECT_EQ(6, arr.GetRawData().size());
}

TEST(Array3Test, ConstructorVector3_test)
{
    Vector3<size_t> arr_size(1, 2, 3);
    Array3<int> arr(arr_size);
    Vector3<size_t> size = arr.GetSize();
    EXPECT_EQ(1, size.x);
    EXPECT_EQ(2, size.y);
    EXPECT_EQ(3, size.z);
    EXPECT_EQ(6, arr.GetRawData().size());
}

TEST(Array3Test, ConstructorCopy_test)
{
    Array3<int> arr(1, 2, 3);
    Array3<int> arr1(arr);
    Vector3<size_t> size = arr1.GetSize();
    EXPECT_EQ(1, size.x);
    EXPECT_EQ(2, size.y);
    EXPECT_EQ(3, size.z);
    EXPECT_EQ(6, arr1.GetRawData().size());
}

TEST(Array3Test, GetSize_test)
{
    Array3<int> arr(1, 2, 3);
    Vector3<size_t> size(1, 2, 3);
    EXPECT_EQ(true, arr.GetSize() == size);
}

TEST(Array3Test, GetRawData_test)
{
    Array3<int> arr(1, 1, 1, 1);
    std::vector<int> vect(1, 1);
    EXPECT_EQ(arr.GetRawData()[0], vect[0]);
}

TEST(Array3Test, GetElementConstSizeT_test)
{
    Array3<int> arr(1, 1, 1, 1);
    EXPECT_EQ(1, arr.GetElement(0, 0, 0));
}

TEST(Array3Test, GetElementReferenceSizeT_test)
{
    Array3<int> arr(1, 1, 1, 1);
    arr.GetElement(0, 0, 0) = 5;
    EXPECT_EQ(5, arr.GetElement(0, 0, 0));
}

TEST(Array3Test, GetElementConstVector3_test)
{
    Array3<int> arr(1, 1, 1, 1);
    Vector3<size_t> pos(0, 0, 0);
    EXPECT_EQ(1, arr.GetElement(pos));
}

TEST(Array3Test, GetElementReferenceVector3_test)
{
    Array3<int> arr(1, 1, 1, 1);
    Vector3<size_t> pos(0, 0, 0);
    arr.GetElement(pos) = 5;
    EXPECT_EQ(5, arr.GetElement(pos));
}

TEST(Array3Test, SetElement_test)
{
    Array3<int> arr(1, 1, 1, 1);
    arr.SetElement(0, 0, 0, 5);
    EXPECT_EQ(5, arr.GetElement(0, 0, 0));
}

TEST(Array3Test, IsEqual_test)
{
    Array3<int> arr(2, 3, 4, 5);
    Array3<int> arr1(2, 3, 4, 5);
    EXPECT_EQ(true, arr.IsEqual(arr1));
}

TEST(Array3Test, FillValue_test)
{
    Vector3<size_t> size(20, 20, 20);
    Array3<int> arr(size, 0);
    auto start = std::chrono::steady_clock::now();
    std::cout << "Fill: ";
    arr.Fill(5);
    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0 << " [s]\n";

    EXPECT_EQ(5, arr.GetElement(0, 0, 0));
    EXPECT_EQ(5, arr.GetElement(1, 0, 0));
    EXPECT_EQ(5, arr.GetElement(0, 1, 0));
    EXPECT_EQ(5, arr.GetElement(1, 1, 0));
    EXPECT_EQ(5, arr.GetElement(0, 0, 1));
    EXPECT_EQ(5, arr.GetElement(1, 0, 1));
    EXPECT_EQ(5, arr.GetElement(0, 1, 1));
    EXPECT_EQ(5, arr.GetElement(1, 1, 1));
}

TEST(Array3Test, ParallelFillValue_test)
{
    Vector3<size_t> size(20, 20, 20);
    Array3<int> arr(size, 0);
    auto start = std::chrono::steady_clock::now();
    std::cout << "ParallelFill: ";
    arr.ParallelFill(5);
    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000000.0 << " [s]\n";
    EXPECT_EQ(5, arr.GetElement(0, 0, 0));
    EXPECT_EQ(5, arr.GetElement(1, 0, 0));
    EXPECT_EQ(5, arr.GetElement(0, 1, 0));
    EXPECT_EQ(5, arr.GetElement(1, 1, 0));
    EXPECT_EQ(5, arr.GetElement(0, 0, 1));
    EXPECT_EQ(5, arr.GetElement(1, 0, 1));
    EXPECT_EQ(5, arr.GetElement(0, 1, 1));
    EXPECT_EQ(5, arr.GetElement(1, 1, 1));
}

TEST(Array3Test, FillArray_test)
{
    Array3<int> arr(2, 2, 1, 1);
    Array3<int> arr1(2, 2, 1, 5);
    arr.Fill(arr1);
    EXPECT_EQ(5, arr.GetElement(0, 0, 0));
    EXPECT_EQ(5, arr.GetElement(1, 0, 0));
    EXPECT_EQ(5, arr.GetElement(0, 1, 0));
    EXPECT_EQ(5, arr.GetElement(1, 1, 0));
}

TEST(Array3Test, Swap_test)
{
    Array3<int> arr(2, 2, 1, 1);
    Array3<int> arr1(2, 2, 1, 5);
    arr.Swap(arr1);
    EXPECT_EQ(5, arr.GetElement(0, 0, 0));
    EXPECT_EQ(5, arr.GetElement(1, 0, 0));
    EXPECT_EQ(5, arr.GetElement(0, 1, 0));
    EXPECT_EQ(5, arr.GetElement(1, 1, 0));
    EXPECT_EQ(1, arr1.GetElement(0, 0, 0));
    EXPECT_EQ(1, arr1.GetElement(1, 0, 0));
    EXPECT_EQ(1, arr1.GetElement(0, 1, 0));
    EXPECT_EQ(1, arr1.GetElement(1, 1, 0));
}

TEST(Array3Test, ResizeVector3_test)
{
    Array3<int> arr(1, 1, 1, 1);
    Vector3<size_t> size(2, 2, 1);
    arr.Resize(size);
    auto arr_size = arr.GetSize();
    EXPECT_EQ(4, arr_size.x * arr_size.y * arr_size.z);
    EXPECT_EQ(1, arr.GetElement(0, 0, 0));
    EXPECT_EQ(0, arr.GetElement(1, 0, 0));
    EXPECT_EQ(0, arr.GetElement(0, 1, 0));
    EXPECT_EQ(0, arr.GetElement(1, 1, 0));
}

TEST(Array3Test, ResizeSizeT_test)
{
    Array3<int> arr(1, 1, 1, 1);
    arr.Resize(2, 2, 1);
    auto arr_size = arr.GetSize();
    EXPECT_EQ(4, arr_size.x * arr_size.y * arr_size.z);
    EXPECT_EQ(1, arr.GetElement(0, 0, 0));
    EXPECT_EQ(0, arr.GetElement(1, 0, 0));
    EXPECT_EQ(0, arr.GetElement(0, 1, 0));
    EXPECT_EQ(0, arr.GetElement(1, 1, 0));
}

TEST(Array3Test, Copy_test)
{
    Array3<int> arr(1, 1, 1, 1);
    Array3<int> arr1(2, 2, 2, 8);
    arr.Copy(arr1);
    auto arr_size = arr.GetSize();
    EXPECT_EQ(8, arr_size.x * arr_size.y * arr_size.z);
    EXPECT_EQ(8, arr.GetElement(0, 0, 0));
    EXPECT_EQ(8, arr.GetElement(1, 0, 0));
    EXPECT_EQ(8, arr.GetElement(0, 1, 0));
    EXPECT_EQ(8, arr.GetElement(1, 1, 0));
    EXPECT_EQ(8, arr.GetElement(0, 0, 1));
    EXPECT_EQ(8, arr.GetElement(1, 0, 1));
    EXPECT_EQ(8, arr.GetElement(0, 1, 1));
    EXPECT_EQ(8, arr.GetElement(1, 1, 1));
}

TEST(Array3Test, Clear_test)
{
    Array3<int> arr(1, 1, 1, 1);
    arr.Clear();
    auto arr_size = arr.GetSize();
    EXPECT_EQ(0, arr_size.x * arr_size.y * arr_size.z);
    EXPECT_EQ(0, arr.GetRawData().size());
}

TEST(Array3Test, EqualEqual_test)
{
    Array3<int> arr(1, 2, 3, 4);
    Array3<int> arr1(1, 2, 3, 4);
    EXPECT_EQ(true, arr == arr1);
}

TEST(Array3Test, NotEqual_test)
{
    Array3<int> arr(1, 2, 3, 4);
    Array3<int> arr1(2, 2, 3, 4);
    EXPECT_EQ(true, arr != arr1);
}

TEST(Array3Test, Assignemnt_test)
{
    Array3<int> arr(1, 1, 1, 1);
    Array3<int> arr1(2, 3, 4, 5);
    arr =arr1;
    EXPECT_EQ(true, arr == arr1);
}

TEST(Array3Test, GetElementBracesSizeT_test)
{
    Array3<int> arr(2, 2, 3, 0);
    EXPECT_EQ(0, arr(0, 1, 2));
}

TEST(Array3Test, GetElementSizeOne_test)
{
    Array3<int> arr(1, 1, 1, 0);
    EXPECT_EQ(0, arr(0, 0, 0));
}

TEST(Array3Test, SetElementBracesVector3_test)
{
    Array3<int> arr(2, 2, 3, 0);
    Vector3<size_t> pos(1, 1, 2);
    arr(pos) = 5;
    EXPECT_EQ(5, arr(pos));
}

TEST(Array3Test, GetElementBracesVector3_test)
{
    Array3<int> arr(2, 2, 3, 0);
    Vector3<size_t> pos(1, 1, 2);
    EXPECT_EQ(0, arr(pos));
}

TEST(Array3Test, SetElementBracesSizeT_test)
{
    Array3<int> arr(2, 2, 3, 0);
    arr(1, 1, 2) = 5;
    EXPECT_EQ(5, arr(1, 1, 2));
}

TEST(Array3Test, SquareBracesGet_test)
{
    Array3<int> arr(1, 2, 3, 4);
    EXPECT_EQ(4, arr[2]);
}

TEST(Array3Test, SquareBracesSet_test)
{
    Array3<int> arr(1, 2, 3, 4);
    arr[3] = 8;
    EXPECT_EQ(8, arr[3]);
}

TEST(Array3Test, ParallelForEachIndex_test)
{
    int factor = 5;
    Vector3<size_t> size(5, 5, 5);
    Array3<int> arr(size, 1);
    arr.ParallelForEachIndex([&](size_t i, size_t j, size_t k)
    {
        arr(i, j, k) = arr(i, j, k) * factor;
    });
    EXPECT_EQ(5, arr(0, 0, 1));
}