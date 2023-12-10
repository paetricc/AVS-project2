/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  TOMÁŠ BÁRTŮ <xbartu11@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    09-12-2023
 **/

#include <iostream>
#include <cmath>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    unsigned totalTriangles = 0;

    #pragma omp parallel default(none) shared(field, totalTriangles)
    {
        #pragma omp single nowait
        {
            totalTriangles = octree(Vec3_t<float>{0.0f, 0.0f, 0.0f}, (float) mGridSize, field);
        }
    }

    return totalTriangles;
}

void TreeMeshBuilder::computeCenterPosition(const Vec3_t<float> &currentPosition, Vec3_t<float> &centerPosition, const float gridSize)
{
    centerPosition.x = (currentPosition.x + gridSize) * mGridResolution;
    centerPosition.y = (currentPosition.y + gridSize) * mGridResolution;
    centerPosition.z = (currentPosition.z + gridSize) * mGridResolution;
}

void TreeMeshBuilder::computeNewPosition(const Vec3_t<float> &currentPosition, Vec3_t<float> &newPosition, const float gridSize, const int i)
{
    newPosition.x = currentPosition.x + gridSize * sc_vertexNormPos[i].x;
    newPosition.y = currentPosition.y + gridSize * sc_vertexNormPos[i].y;
    newPosition.z = currentPosition.z + gridSize * sc_vertexNormPos[i].z;
}

bool TreeMeshBuilder::isBlockEmpty(const Vec3_t<float> currentPosition, const ParametricScalarField& field, const float gridSize, const float GridResolution)
{
    Vec3_t<float> newPosition;
    computeCenterPosition(currentPosition, newPosition, gridSize);
    return ((evaluateFieldAt(newPosition, field)) > (mIsoLevel + SQRT3 * 0.5f * GridResolution));
}

unsigned TreeMeshBuilder::octree(const Vec3_t<float> &currentPosition, float gridSize,  const ParametricScalarField &field)
{
    unsigned triangles = 0;
    float halfOfGridSize = gridSize * 0.5f;

    if(isBlockEmpty(currentPosition, field, halfOfGridSize, (float)gridSize * mGridResolution))
    {
        return 0;
    }

    if(gridSize <= MIN_GRID)
    {
        return buildCube(currentPosition, field);
    }

    for (int i = 0; i < 8; i++)
    {
        #pragma omp task default(none) firstprivate(i, currentPosition, halfOfGridSize) shared(triangles, field)
        {
            Vec3_t<float> newPosition;
            computeNewPosition(currentPosition, newPosition, halfOfGridSize, i);

            #pragma omp atomic
            triangles += octree(newPosition, halfOfGridSize, field);
        }
    }

    #pragma omp taskwait
    {
        return triangles;
    }
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    // NOTE: This method is called from "buildCube(...)"!

    // 1. Store pointer to and number of 3D points in the field
    //    (to avoid "data()" and "size()" call in the loop).
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    // 2. Find minimum square distance from points "pos" to any point in the
    //    field.
    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // 3. Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    // NOTE: This method is called from "buildCube(...)"!

    // Store generated triangle into vector (array) of generated triangles.
    // The pointer to data in this array is return by "getTrianglesArray(...)" call
    // after "marchCubes(...)" call ends.
    #pragma omp critical
    {
        mTriangles.push_back(triangle);
    }
}