//
// Created by KYL.ai on 2021-03-17.
//

#ifndef FACETOOL_POSTPROCESS_H
#define FACETOOL_POSTPROCESS_H

#include <vector>
#include <queue>
#include <limits>
#include <cmath>
#include <numeric>

/**
 * Method sorts probabilty values and returns k top scores with index as vector
 * @param values : sofmax probabilty value pointer;
 * @param len : length of the pointer
 * @param k : k values
 * @param results : vector of probabilty value and index
 */

// Function to print the
// index of an element
int getIndex(std::vector<float> v, float K)
{
    auto it = std::find(v.begin(), v.end(), K);
    int index = -1;

    if (it != v.end()) // K랑 일치하는 벡터가 있다면
    {
        index = it - v.begin();
    }
    else
    { // K랑 일치하는 벡터가 없다면
        index = 0;
    }

    return index;
}

template <typename T>
std::deque<size_t> sortIndexes(const std::vector<T> &v)
{
    std::deque<size_t> indices(v.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::stable_sort(std::begin(indices), std::end(indices), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

    return indices;
}

std::vector<uint64_t> nms(const std::vector<std::array<float, 4>> &bboxes,           //
                          const std::vector<float> &scores,                          //
                          const float overlapThresh = 0.45,                          //
                          const uint64_t topK = std::numeric_limits<uint64_t>::max() //
)
{
    assert(bboxes.size() > 0);
    uint64_t boxesLength = bboxes.size();
    const uint64_t realK = std::max(std::min(boxesLength, topK), static_cast<uint64_t>(1));

    std::vector<uint64_t> keepIndices;
    keepIndices.reserve(realK);

    std::deque<uint64_t> sortedIndices = sortIndexes(scores);

    // keep only topk bboxes
    for (uint64_t i = 0; i < boxesLength - realK; ++i)
    {
        sortedIndices.pop_front();
    }

    std::vector<float> areas;
    areas.reserve(boxesLength);
    std::transform(std::begin(bboxes), std::end(bboxes), std::back_inserter(areas),
                   [](const auto &elem) { return (elem[2] - elem[0]) * (elem[3] - elem[1]); });

    while (!sortedIndices.empty())
    {
        uint64_t currentIdx = sortedIndices.back();
        keepIndices.emplace_back(currentIdx);

        if (sortedIndices.size() == 1)
        {
            break;
        }

        sortedIndices.pop_back();
        std::vector<float> ious;
        ious.reserve(sortedIndices.size());

        const auto &curBbox = bboxes[currentIdx];
        const float curArea = areas[currentIdx];

        std::deque<uint64_t> newSortedIndices;

        for (const uint64_t elem : sortedIndices)
        {
            const auto &bbox = bboxes[elem];
            float tmpXmin = std::max(curBbox[0], bbox[0]);
            float tmpYmin = std::max(curBbox[1], bbox[1]);
            float tmpXmax = std::min(curBbox[2], bbox[2]);
            float tmpYmax = std::min(curBbox[3], bbox[3]);

            float tmpW = std::max<float>(tmpXmax - tmpXmin, 0.0);
            float tmpH = std::max<float>(tmpYmax - tmpYmin, 0.0);

            const float intersection = tmpW * tmpH;
            const float tmpArea = areas[elem];
            const float unionArea = tmpArea + curArea - intersection;
            const float iou = intersection / unionArea;

            if (iou <= overlapThresh)
            {
                newSortedIndices.emplace_back(elem);
            }
        }

        sortedIndices = newSortedIndices;
    }

    return keepIndices;
}

#endif //FACETOOL_POSTPROCESS_H
