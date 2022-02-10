#ifndef FRAME_HPP
#define FRAME_HPP

class Frame
{
    public:
        Frame();
        Frame(double timeIntervalInSeconds);
        
        void Advance();
        void Advance(int number_of_frames);

        Frame& operator++();
        Frame operator++(int);

        unsigned long int GetIndex() const;
        double GetTimeIntervalInSeconds() const;
        double GetTimeInSeconds() const;

    private:
        unsigned long int _index = 0;
        double _timeIntervalInSeconds = 1/60.0f;
};

#endif // FRAME_HPP