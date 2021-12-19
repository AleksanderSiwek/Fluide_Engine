#include <gtest/gtest.h>
#include "../src/test_hdr.hpp"

TEST(ExampleTests_2, add_one)
{
    EXPECT_EQ(2, add_one(1));
}

TEST(ExampleTests_4, add_one_ok)
{
    EXPECT_EQ(3, add_one(2));
}