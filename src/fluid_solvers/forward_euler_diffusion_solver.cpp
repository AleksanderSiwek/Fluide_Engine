#include "forward_euler_diffusion_solver.hpp"

ForwardEulerDiffusionSolver::ForwardEulerDiffusionSolver()
{

}

ForwardEulerDiffusionSolver::~ForwardEulerDiffusionSolver()
{

}

void ForwardEulerDiffusionSolver::CalculateDiffusion(const FaceCenteredGrid3D& source_grid, const FluidMarkers& fluidMarkers, double timeIntervalInSeconds, FaceCenteredGrid3D* output)
{
    Vector3<size_t> size = source_grid.GetSize();
    output->Resize(size);
    
    const Array3<double>& xData = source_grid.GetDataXRef();
    const Array3<double>& yData = source_grid.GetDataYRef();
    const Array3<double>& zData = source_grid.GetDataZRef();

    for(size_t i = 0; i < size.x; i++)
    {
        for(size_t j = 0; j < size.y; j++) 
        {
            for(size_t k = 0; k < size.z; k++)
            {
                output->x(i, j, k) = source_grid.x(i, j, k) + _viscosity * timeIntervalInSeconds * CalculateLaplacian(xData, fluidMarkers, source_grid.GetGridSpacing(), i, j, k);
                output->y(i, j, k) = source_grid.y(i, j, k) + _viscosity * timeIntervalInSeconds * CalculateLaplacian(yData, fluidMarkers, source_grid.GetGridSpacing(), i, j, k);
                output->z(i, j, k) = source_grid.z(i, j, k) + _viscosity * timeIntervalInSeconds * CalculateLaplacian(zData, fluidMarkers, source_grid.GetGridSpacing(), i, j, k);
            }            
        }       
    }
}

double ForwardEulerDiffusionSolver::CalculateLaplacian(const Array3<double>& grid, const FluidMarkers& fluidMarkers, Vector3<double> gridSpacing, size_t i, size_t j, size_t k)
{
    Vector3<size_t> size = grid.GetSize();

    if(!(i < size.x && j < size.y && k < size.z))
    {
        return 0;
    }

    double center = grid(i, j, k);
    double dleft = 0;
    double dright = 0;
    double ddown = 0;
    double dup = 0;
    double dback = 0;
    double dfront = 0;

    if(i > 0 && fluidMarkers(i - 1, j, k) == FLUID_MARK)
        dleft = center - grid(i - 1, j, k);
    if(i  <= grid.GetSize().x && fluidMarkers(i + 1, j, k) == FLUID_MARK)
        dright = center - grid(i + 1, j, k);
    if(j > 0 && fluidMarkers(i, j - 1, k) == FLUID_MARK)
        ddown = center - grid(i , j - 1, k);
    if(j  <= grid.GetSize().y && fluidMarkers(i, j + 1, k) == FLUID_MARK)
        dup = center - grid(i, j + 1, k);
    if(k > 0 && fluidMarkers(i, j, k - 1) == FLUID_MARK)
        dback = center - grid(i, j, k - 1);
    if(k <= grid.GetSize().z && fluidMarkers(i, j, k + 1) == FLUID_MARK)
        dfront = center - grid(i, j, k + 1);

    auto gridSpacingSuqared = gridSpacing * gridSpacing;
    return (dright - dleft)/(gridSpacingSuqared.x) + (dup - dright)/(gridSpacingSuqared.y) + (dfront - dback)/(gridSpacingSuqared.z);
}
