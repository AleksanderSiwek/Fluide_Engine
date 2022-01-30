#include "frame.hpp"

Frame::Frame()
{
    
}

Frame::Frame(double timeIntervalInSeconds)
{
    _timeIntervalInSeconds = timeIntervalInSeconds;
}

void Frame::Advance()
{
    _index++;
}

void Frame::Advance(int number_of_frames)
{
    _index += number_of_frames;
}

Frame& Frame::operator++()
{
    Advance();
    return *this;
}

Frame Frame::operator++(int i)
{
    Frame result = *this;
    Advance();
    return result;
}

unsigned long int Frame::GetIndex() const
{
    return _index;
}

double Frame::GetTimeIntervalInSeconds() const
{
    return _timeIntervalInSeconds;
}

double Frame::GetTimeInSeconds() const
{
    return _index * _timeIntervalInSeconds;
}