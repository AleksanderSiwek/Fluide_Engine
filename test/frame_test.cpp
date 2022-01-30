#include <gtest/gtest.h>
#include "../src/frame.hpp"


TEST(FrameTest, Constructor_test)
{
    Frame frame;
    EXPECT_EQ(0, frame.GetIndex());
}

TEST(FrameTest, ConstructorWithTimeInterval_test)
{
    Frame frame(1);
    EXPECT_EQ(1, frame.GetTimeIntervalInSeconds());
}

TEST(FrameTest, GetIndex_test)
{
    Frame frame;
    EXPECT_EQ(0, frame.GetIndex());
}

TEST(FrameTest, GetTimeIntervalInSeconds_test)
{
    Frame frame;
    EXPECT_EQ(1/60.0f, frame.GetTimeIntervalInSeconds());
}

TEST(FrameTest, GetTimeInSeconds_test)
{
    Frame frame;
    frame++;
    EXPECT_EQ(1/60.0f, frame.GetTimeInSeconds());
}

TEST(FrameTest, AdvanceFrame_test)
{
    Frame frame;
    frame.Advance();
    EXPECT_EQ(1, frame.GetIndex());
}

TEST(FrameTest, AdvanceNumberOfFrames_test)
{
    Frame frame;
    int number_of_frames = 5;
    frame.Advance(number_of_frames);
    EXPECT_EQ(5, frame.GetIndex());
}

TEST(FrameTest, AdvanceFrameByPPOperator_test)
{
    Frame frame;
    frame++;
    EXPECT_EQ(1, frame.GetIndex());
}

