#include "forward_euler_diffusion_solver.hpp"

ForwardEulerDiffusionSolver::ForwardEulerDiffusionSolver()
{

}

ForwardEulerDiffusionSolver::~ForwardEulerDiffusionSolver()
{

}

FaceCenteredGrid3D ForwardEulerDiffusionSolver::CalculateDiffusion(FaceCenteredGrid3D& source_grid, const Array3<uint8_t>& fluidMarkers, double timeIntervalInSeconds)
{
    FaceCenteredGrid3D output = FaceCenteredGrid3D(source_grid.GetSize(), source_grid.GetOrigin(), source_grid.GetGridSpacing());

    Array3<double>* xData = source_grid.GetDataXPtr();
    Array3<double>* yData = source_grid.GetDataYPtr();
    Array3<double>* zData = source_grid.GetDataZPtr();

    for(size_t i = 0; i < source_grid.GetSize().x; i++)
    {
        for(size_t j = 0; j < source_grid.GetSize().y; j++) 
        {
            for(size_t k = 0; k < source_grid.GetSize().z; k++)
            {
                output.x(i, j, k) = source_grid.x(i, j, k) + _viscosity * timeIntervalInSeconds * CalculateLaplacian(xData, fluidMarkers, source_grid.GetGridSpacing(), i, j, k);
                output.y(i, j, k) = source_grid.y(i, j, k) + _viscosity * timeIntervalInSeconds * CalculateLaplacian(yData, fluidMarkers, source_grid.GetGridSpacing(), i, j, k);
                output.z(i, j, k) = source_grid.z(i, j, k) + _viscosity * timeIntervalInSeconds * CalculateLaplacian(zData, fluidMarkers, source_grid.GetGridSpacing(), i, j, k);
            }            
        }       
    }

    return output;
}

void ForwardEulerDiffusionSolver::CalculateDiffusion(FaceCenteredGrid3D& source_grid, const Array3<uint8_t>& fluidMarkers, double timeIntervalInSeconds, FaceCenteredGrid3D* dest_grid)
{
    FaceCenteredGrid3D* out = new FaceCenteredGrid3D();
    dest_grid = out;
}

double ForwardEulerDiffusionSolver::CalculateLaplacian(Array3<double>* grid, const Array3<uint8_t> fluidMarkers, Vector3<double> gridSpacing, size_t i, size_t j, size_t k)
{
    if(!(i < grid->GetSize().x && j < grid->GetSize().y && k < grid->GetSize().z))
    {
        return 0;
    }

    double center = (*grid)(i, j, k);
    double dleft = 0;
    double dright = 0;
    double ddown = 0;
    double dup = 0;
    double dback = 0;
    double dfront = 0;

    // 0 - air, 1 - fluid TO DO change fluid markers to something prettier
    if(i > 0 && fluidMarkers(i - 1, j, k) == 1)
        dleft = center - (*grid)(i - 1, j, k);
    if(i  <= grid->GetSize().x && fluidMarkers(i + 1, j, k) == 1)
        dright = center - (*grid)(i + 1, j, k);
    if(j > 0 && fluidMarkers(i, j - 1, k) == 1)
        ddown = center - (*grid)(i , j - 1, k);
    if(j  <= grid->GetSize().y && fluidMarkers(i, j + 1, k) == 1)
        dup = center - (*grid)(i, j + 1, k);
    if(k > 0 && fluidMarkers(i, j, k - 1) == 1)
        dback = center - (*grid)(i, j, k - 1);
    if(k <= grid->GetSize().z && fluidMarkers(i, j, k + 1) == 1)
        dfront = center - (*grid)(i, j, k + 1);

    auto gridSpacingSuqared = gridSpacing * gridSpacing;
    return (dright - dleft)/(gridSpacingSuqared.x) + (dup - dright)/(gridSpacingSuqared.y) + (dfront - dback)/(gridSpacingSuqared.z);
}
