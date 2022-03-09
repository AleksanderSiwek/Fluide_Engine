#ifndef _TRIANGLE_MESH_HPP
#define _TRIANGLE_MESH_HPP

class TriangleMesh
{
    public:
        TriangleMesh();
        ~TriangleMesh();

    private:
        std::vector<Vector3<double>> vertices;
};


#endif // _TRIANGLE_MESH_HPP