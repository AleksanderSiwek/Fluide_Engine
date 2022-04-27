#include "forward_euler_diffusion_solver.hpp"

ForwardEulerDiffusionSolver::ForwardEulerDiffusionSolver()
{

}

ForwardEulerDiffusionSolver::~ForwardEulerDiffusionSolver()
{

}

void ForwardEulerDiffusionSolver::Solve(const FaceCenteredGrid3D& sourceGrid, const ScalarGrid3D& fluidSdf, double viscosity, double timeIntervalInSeconds, FaceCenteredGrid3D* output)
{
    Vector3<size_t> size = sourceGrid.GetSize();
    output->Resize(size);
    BuildMarkers(fluidSdf, size, sourceGrid);

    const auto& xData = sourceGrid.GetDataXRef();
    const auto& yData = sourceGrid.GetDataYRef();
    const auto& zData = sourceGrid.GetDataZRef();

    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++) 
        {
            for(size_t k = 0; k < size.z; k++)
            {
                if(_fluidMarkers(i, j, k) == FLUID_MARK)
                {
                    output->x(i, j, k) = sourceGrid.x(i, j, k) + viscosity * timeIntervalInSeconds * CalculateLaplacian(xData, sourceGrid.GetGridSpacing(), i, j, k);
                    output->y(i, j, k) = sourceGrid.y(i, j, k) + viscosity * timeIntervalInSeconds * CalculateLaplacian(yData, sourceGrid.GetGridSpacing(), i, j, k);
                    output->z(i, j, k) = sourceGrid.z(i, j, k) + viscosity * timeIntervalInSeconds * CalculateLaplacian(zData, sourceGrid.GetGridSpacing(), i, j, k);
                }
                else
                {
                    output->x(i, j, k) = sourceGrid.x(i, j, k);
                    output->y(i, j, k) = sourceGrid.y(i, j, k);
                    output->z(i, j, k) = sourceGrid.z(i, j, k);
                }
            }            
        }       
    }
}

double ForwardEulerDiffusionSolver::CalculateLaplacian(const Array3<double>& grid, Vector3<double> gridSpacing, size_t i, size_t j, size_t k)
{
    Vector3<size_t> size = _fluidMarkers.GetSize();

    if(i >= size.x || j >= size.y || k >= size.z)
    {
        return 0;
    }

    int signedI = static_cast<int>(i);
    int signedJ = static_cast<int>(j);
    int signedK = static_cast<int>(k);

    double center = grid(i, j, k);
    double dleft = 0;
    double dright = 0;
    double ddown = 0;
    double dup = 0;
    double dback = 0;
    double dfront = 0;

    if(signedI - 1 >= 0 && _fluidMarkers(i - 1, j, k) == FLUID_MARK)
        dleft = center - grid(i - 1, j, k);
    if(signedI + 1 < size.x && _fluidMarkers(i + 1, j, k) == FLUID_MARK)
        dright = grid(i + 1, j, k) - center;

    if(signedJ - 1 >= 0 && _fluidMarkers(i, j - 1, k) == FLUID_MARK)
        ddown = center - grid(i , j - 1, k);
    if(signedJ + 1 < size.y && _fluidMarkers(i, j + 1, k) == FLUID_MARK)
        dup = grid(i, j + 1, k) - center;

    if(signedK - 1 >= 0 && _fluidMarkers(i, j, k - 1) == FLUID_MARK)
        dback = center - grid(i, j, k - 1);
    if(signedK + 1 < size.z && _fluidMarkers(i, j, k + 1) == FLUID_MARK)
        dfront = grid(i, j, k + 1) - center;

    auto gridSpacingSuqared = gridSpacing * gridSpacing;
    return (dright - dleft)/(gridSpacingSuqared.x) + (dup - ddown)/(gridSpacingSuqared.y) + (dfront - dback)/(gridSpacingSuqared.z);
}

void ForwardEulerDiffusionSolver::BuildMarkers(const ScalarGrid3D& fluidSdf, const Vector3<size_t>& size, const FaceCenteredGrid3D& sourceGrid)
{
    _fluidMarkers.Resize(size);
    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++)
        {
            for(size_t k = 0; k < size.z; k++)
            {
                if(fluidSdf.Sample(sourceGrid.GridIndexToPosition(i, j, k)) < 0)
                {
                    _fluidMarkers(i, j, k) = FLUID_MARK;
                }
                else
                {
                    _fluidMarkers(i, j, k) = AIR_MARK;
                }
            }
        }
    }
}
