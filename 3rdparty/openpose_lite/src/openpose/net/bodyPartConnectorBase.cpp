#include <openpose/utilities/check.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/pose/poseParameters.hpp>
#include <openpose/net/bodyPartConnectorBase.hpp>

namespace op
{
    template <typename T>
    inline T getScoreAB(const int i, const int j, const T* const candidateAPtr, const T* const candidateBPtr,
                        const T* const mapX, const T* const mapY, const Point<int>& heatMapSize,
                        const T interThreshold, const T interMinAboveThreshold)
    {
        try
        {
            const auto vectorAToBX = candidateBPtr[3*j] - candidateAPtr[3*i];
            const auto vectorAToBY = candidateBPtr[3*j+1] - candidateAPtr[3*i+1];
            const auto vectorAToBMax = fastMax(std::abs(vectorAToBX), std::abs(vectorAToBY));
            const auto numberPointsInLine = fastMax(
                5, fastMin(25, positiveIntRound(std::sqrt(5*vectorAToBMax))));
            const auto vectorNorm = T(std::sqrt( vectorAToBX*vectorAToBX + vectorAToBY*vectorAToBY ));
            // If the peaksPtr are coincident. Don't connect them.
            if (vectorNorm > 1e-6)
            {
                const auto sX = candidateAPtr[3*i];
                const auto sY = candidateAPtr[3*i+1];
                const auto vectorAToBNormX = vectorAToBX/vectorNorm;
                const auto vectorAToBNormY = vectorAToBY/vectorNorm;

                auto sum = T(0);
                auto count = 0u;
                const auto vectorAToBXInLine = vectorAToBX/numberPointsInLine;
                const auto vectorAToBYInLine = vectorAToBY/numberPointsInLine;
                for (auto lm = 0; lm < numberPointsInLine; lm++)
                {
                    const auto mX = fastMax(
                        0, fastMin(heatMapSize.x-1, positiveIntRound(sX + lm*vectorAToBXInLine)));
                    const auto mY = fastMax(
                        0, fastMin(heatMapSize.y-1, positiveIntRound(sY + lm*vectorAToBYInLine)));
                    const auto idx = mY * heatMapSize.x + mX;
                    const auto score = (vectorAToBNormX*mapX[idx] + vectorAToBNormY*mapY[idx]);
                    if (score > interThreshold)
                    {
                        sum += score;
                        count++;
                    }
                }
                if (count/T(numberPointsInLine) > interMinAboveThreshold)
                    return sum/count;
            }
            return T(0);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return T(0);
        }
    }

    template <typename T>
    std::vector<std::pair<std::vector<int>, T>> createPeopleVector(
        const T* const heatMapPtr, const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
        const int maxPeaks, const T interThreshold, const T interMinAboveThreshold,
        const std::vector<unsigned int>& bodyPartPairs, const unsigned int numberBodyParts,
        const unsigned int numberBodyPartPairs, const Array<T>& pairScores)
    {
        try
        {
            if (poseModel != PoseModel::BODY_25 && poseModel != PoseModel::COCO_18
                && poseModel != PoseModel::MPI_15 && poseModel != PoseModel::MPI_15_4)
                error("Model not implemented for CPU body connector.", __LINE__, __FUNCTION__, __FILE__);

            // std::vector<std::pair<std::vector<int>, double>> refers to:
            //     - std::vector<int>: [body parts locations, #body parts found]
            //     - double: person subset score
            std::vector<std::pair<std::vector<int>, T>> peopleVector;
            const auto& mapIdx = getPoseMapIndex(poseModel);
            const auto numberBodyPartsAndBkg = numberBodyParts + (addBkgChannel(poseModel) ? 1 : 0);
            const auto vectorSize = numberBodyParts+1;
            const auto peaksOffset = 3*(maxPeaks+1);
            const auto heatMapOffset = heatMapSize.area();
            // Iterate over it PAF connection, e.g., neck-nose, neck-Lshoulder, etc.
            for (auto pairIndex = 0u; pairIndex < numberBodyPartPairs; pairIndex++)
            {
                const auto bodyPartA = bodyPartPairs[2*pairIndex];
                const auto bodyPartB = bodyPartPairs[2*pairIndex+1];
                const auto* candidateAPtr = peaksPtr + bodyPartA*peaksOffset;
                const auto* candidateBPtr = peaksPtr + bodyPartB*peaksOffset;
                const auto numberPeaksA = positiveIntRound(candidateAPtr[0]);
                const auto numberPeaksB = positiveIntRound(candidateBPtr[0]);

                // E.g., neck-nose connection. If one of them is empty (e.g., no noses detected)
                // Add the non-empty elements into the peopleVector
                if (numberPeaksA == 0 || numberPeaksB == 0)
                {
                    // E.g., neck-nose connection. If no necks, add all noses
                    // Change w.r.t. other
                    if (numberPeaksA == 0) // numberPeaksB == 0 or not
                    {
                        // Non-MPI
                        if (numberBodyParts != 15)
                        {
                            for (auto i = 1; i <= numberPeaksB; i++)
                            {
                                bool found = false;
                                for (const auto& personVector : peopleVector)
                                {
                                    const auto off = (int)bodyPartB*peaksOffset + i*3 + 2;
                                    if (personVector.first[bodyPartB] == off)
                                    {
                                        found = true;
                                        break;
                                    }
                                }
                                // Add new personVector with this element
                                if (!found)
                                {
                                    std::vector<int> rowVector(vectorSize, 0);
                                    // Store the index
                                    rowVector[ bodyPartB ] = bodyPartB*peaksOffset + i*3 + 2;
                                    // Last number in each row is the parts number of that person
                                    rowVector.back() = 1;
                                    const auto personScore = candidateBPtr[i*3+2];
                                    // Second last number in each row is the total score
                                    peopleVector.emplace_back(std::make_pair(rowVector, personScore));
                                }
                            }
                        }
                        // MPI
                        else
                        {
                            for (auto i = 1; i <= numberPeaksB; i++)
                            {
                                std::vector<int> rowVector(vectorSize, 0);
                                // Store the index
                                rowVector[ bodyPartB ] = bodyPartB*peaksOffset + i*3 + 2;
                                // Last number in each row is the parts number of that person
                                rowVector.back() = 1;
                                // Second last number in each row is the total score
                                const auto personScore = candidateBPtr[i*3+2];
                                peopleVector.emplace_back(std::make_pair(rowVector, personScore));
                            }
                        }
                    }
                    // E.g., neck-nose connection. If no noses, add all necks
                    else // if (numberPeaksA != 0 && numberPeaksB == 0)
                    {
                        // Non-MPI
                        if (numberBodyParts != 15)
                        {
                            for (auto i = 1; i <= numberPeaksA; i++)
                            {
                                bool found = false;
                                const auto indexA = bodyPartA;
                                for (const auto& personVector : peopleVector)
                                {
                                    const auto off = (int)bodyPartA*peaksOffset + i*3 + 2;
                                    if (personVector.first[indexA] == off)
                                    {
                                        found = true;
                                        break;
                                    }
                                }
                                if (!found)
                                {
                                    std::vector<int> rowVector(vectorSize, 0);
                                    // Store the index
                                    rowVector[ bodyPartA ] = bodyPartA*peaksOffset + i*3 + 2;
                                    // Last number in each row is the parts number of that person
                                    rowVector.back() = 1;
                                    // Second last number in each row is the total score
                                    const auto personScore = candidateAPtr[i*3+2];
                                    peopleVector.emplace_back(std::make_pair(rowVector, personScore));
                                }
                            }
                        }
                        // MPI
                        else
                        {
                            for (auto i = 1; i <= numberPeaksA; i++)
                            {
                                std::vector<int> rowVector(vectorSize, 0);
                                // Store the index
                                rowVector[ bodyPartA ] = bodyPartA*peaksOffset + i*3 + 2;
                                // Last number in each row is the parts number of that person
                                rowVector.back() = 1;
                                // Second last number in each row is the total score
                                const auto personScore = candidateAPtr[i*3+2];
                                peopleVector.emplace_back(std::make_pair(rowVector, personScore));
                            }
                        }
                    }
                }
                // E.g., neck-nose connection. If necks and noses, look for maximums
                else // if (numberPeaksA != 0 && numberPeaksB != 0)
                {
                    // (score, indexA, indexB). Inverted order for easy std::sort
                    std::vector<std::tuple<double, int, int>> allABConnections;
                    // Note: Problem of this function, if no right PAF between A and B, both elements are
                    // discarded. However, they should be added indepently, not discarded
                    if (heatMapPtr != nullptr)
                    {
                        const auto* mapX = heatMapPtr
                                         + (numberBodyPartsAndBkg + mapIdx[2*pairIndex]) * heatMapOffset;
                        const auto* mapY = heatMapPtr
                                         + (numberBodyPartsAndBkg + mapIdx[2*pairIndex+1]) * heatMapOffset;
                        // E.g., neck-nose connection. For each neck
                        for (auto i = 1; i <= numberPeaksA; i++)
                        {
                            // E.g., neck-nose connection. For each nose
                            for (auto j = 1; j <= numberPeaksB; j++)
                            {
                                // Initial PAF
                                auto scoreAB = getScoreAB(i, j, candidateAPtr, candidateBPtr, mapX, mapY,
                                                          heatMapSize, interThreshold, interMinAboveThreshold);

                                // E.g., neck-nose connection. If possible PAF between neck i, nose j --> add
                                // parts score + connection score
                                if (scoreAB > 1e-6)
                                    allABConnections.emplace_back(std::make_tuple(scoreAB, i, j));
                            }
                        }
                    }
                    else if (!pairScores.empty())
                    {
                        const auto firstIndex = (int)pairIndex*pairScores.getSize(1)*pairScores.getSize(2);
                        // E.g., neck-nose connection. For each neck
                        for (auto i = 0; i < numberPeaksA; i++)
                        {
                            const auto iIndex = firstIndex + i*pairScores.getSize(2);
                            // E.g., neck-nose connection. For each nose
                            for (auto j = 0; j < numberPeaksB; j++)
                            {
                                const auto scoreAB = pairScores[iIndex + j];

                                // E.g., neck-nose connection. If possible PAF between neck i, nose j --> add
                                // parts score + connection score
                                if (scoreAB > 1e-6)
                                    // +1 because peaksPtr starts with counter
                                    allABConnections.emplace_back(std::make_tuple(scoreAB, i+1, j+1));
                            }
                        }
                    }
                    else
                        error("Error. Should not reach here.", __LINE__, __FUNCTION__, __FILE__);

                    // select the top minAB connection, assuming that each part occur only once
                    // sort rows in descending order based on parts + connection score
                    if (!allABConnections.empty())
                        std::sort(allABConnections.begin(), allABConnections.end(),
                                  std::greater<std::tuple<double, int, int>>());

                    std::vector<std::tuple<int, int, double>> abConnections; // (x, y, score)
                    {
                        const auto minAB = fastMin(numberPeaksA, numberPeaksB);
                        std::vector<int> occurA(numberPeaksA, 0);
                        std::vector<int> occurB(numberPeaksB, 0);
                        auto counter = 0;
                        for (const auto& aBConnection : allABConnections)
                        {
                            const auto score = std::get<0>(aBConnection);
                            const auto indexA = std::get<1>(aBConnection);
                            const auto indexB = std::get<2>(aBConnection);
                            if (!occurA[indexA-1] && !occurB[indexB-1])
                            {
                                abConnections.emplace_back(std::make_tuple(bodyPartA*peaksOffset + indexA*3 + 2,
                                                                           bodyPartB*peaksOffset + indexB*3 + 2,
                                                                           score));
                                counter++;
                                if (counter==minAB)
                                    break;
                                occurA[indexA-1] = 1;
                                occurB[indexB-1] = 1;
                            }
                        }
                    }

                    // Cluster all the body part candidates into peopleVector based on the part connection
                    if (!abConnections.empty())
                    {
                        // initialize first body part connection 15&16
                        if (pairIndex==0)
                        {
                            for (const auto& abConnection : abConnections)
                            {
                                std::vector<int> rowVector(numberBodyParts+3, 0);
                                const auto indexA = std::get<0>(abConnection);
                                const auto indexB = std::get<1>(abConnection);
                                const auto score = std::get<2>(abConnection);
                                rowVector[bodyPartPairs[0]] = indexA;
                                rowVector[bodyPartPairs[1]] = indexB;
                                rowVector.back() = 2;
                                // add the score of parts and the connection
                                const auto personScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                                peopleVector.emplace_back(std::make_pair(rowVector, personScore));
                            }
                        }
                        // Add ears connections (in case person is looking to opposite direction to camera)
                        // Note: This has some issues:
                        //     - It does not prevent repeating the same keypoint in different people
                        //     - Assuming I have nose,eye,ear as 1 person subset, and whole arm as another one, it will not
                        //       merge them both
                        else if (
                            (numberBodyParts == 18 && (pairIndex==17 || pairIndex==18))
                            || ((numberBodyParts == 19 || (numberBodyParts == 25)
                                 || numberBodyParts == 59 || numberBodyParts == 65)
                                && (pairIndex==18 || pairIndex==19))
                            )
                        {
                            for (const auto& abConnection : abConnections)
                            {
                                const auto indexA = std::get<0>(abConnection);
                                const auto indexB = std::get<1>(abConnection);
                                for (auto& personVector : peopleVector)
                                {
                                    auto& personVectorA = personVector.first[bodyPartA];
                                    auto& personVectorB = personVector.first[bodyPartB];
                                    if (personVectorA == indexA && personVectorB == 0)
                                    {
                                        personVectorB = indexB;
                                        // // This seems to harm acc 0.1% for BODY_25
                                        // personVector.first.back()++;
                                    }
                                    else if (personVectorB == indexB && personVectorA == 0)
                                    {
                                        personVectorA = indexA;
                                        // // This seems to harm acc 0.1% for BODY_25
                                        // personVector.first.back()++;
                                    }
                                }
                            }
                        }
                        else
                        {
                            // A is already in the peopleVector, find its connection B
                            for (const auto& abConnection : abConnections)
                            {
                                const auto indexA = std::get<0>(abConnection);
                                const auto indexB = std::get<1>(abConnection);
                                const auto score = T(std::get<2>(abConnection));
                                bool found = false;
                                for (auto& personVector : peopleVector)
                                {
                                    // Found partA in a peopleVector, add partB to same one.
                                    if (personVector.first[bodyPartA] == indexA)
                                    {
                                        personVector.first[bodyPartB] = indexB;
                                        personVector.first.back()++;
                                        personVector.second += peaksPtr[indexB] + score;
                                        found = true;
                                        break;
                                    }
                                }
                                // Not found partA in peopleVector, add new peopleVector element
                                if (!found)
                                {
                                    std::vector<int> rowVector(vectorSize, 0);
                                    rowVector[bodyPartA] = indexA;
                                    rowVector[bodyPartB] = indexB;
                                    rowVector.back() = 2;
                                    const auto personScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                                    peopleVector.emplace_back(std::make_pair(rowVector, personScore));
                                }
                            }
                        }
                    }
                }
            }
            return peopleVector;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    template <typename T>
    void removePeopleBelowThresholds(
        std::vector<int>& validSubsetIndexes, int& numberPeople,
        const std::vector<std::pair<std::vector<int>, T>>& peopleVector, const unsigned int numberBodyParts,
        const int minSubsetCnt, const T minSubsetScore, const int maxPeaks, const bool maximizePositives)
    {
        try
        {
            // Delete people below the following thresholds:
                // a) minSubsetCnt: removed if less than minSubsetCnt body parts
                // b) minSubsetScore: removed if global score smaller than this
                // c) maxPeaks (POSE_MAX_PEOPLE): keep first maxPeaks people above thresholds
            numberPeople = 0;
            validSubsetIndexes.clear();
            validSubsetIndexes.reserve(fastMin((size_t)maxPeaks, peopleVector.size()));
            for (auto index = 0u ; index < peopleVector.size() ; index++)
            {
                auto personCounter = peopleVector[index].first.back();
                // Foot keypoints do not affect personCounter (too many false positives,
                // same foot usually appears as both left and right keypoints)
                // Pros: Removed tons of false positives
                // Cons: Standalone leg will never be recorded
                if (!maximizePositives && (numberBodyParts == 25 || numberBodyParts > 70))
                {
                    // No consider foot keypoints for that
                    for (auto i = 19 ; i < 25 ; i++)
                        personCounter -= (peopleVector[index].first.at(i) > 0);
                    // No consider hand keypoints for that
                    if (numberBodyParts > 70)
                        for (auto i = 25 ; i < 65 ; i++)
                            personCounter -= (peopleVector[index].first.at(i) > 0);
                }
                const auto personScore = peopleVector[index].second;
                if (personCounter >= minSubsetCnt && (personScore/personCounter) >= minSubsetScore)
                {
                    numberPeople++;
                    validSubsetIndexes.emplace_back(index);
                    if (numberPeople == maxPeaks)
                        break;
                }
                else if ((personCounter < 1 && numberBodyParts != 25 && numberBodyParts < 70) || personCounter < 0)
                    error("Bad personCounter (" + std::to_string(personCounter) + "). Bug in this"
                          " function if this happens.", __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void peopleVectorToPeopleArray(Array<T>& poseKeypoints, Array<T>& poseScores, const T scaleFactor,
                                   const std::vector<std::pair<std::vector<int>, T>>& peopleVector,
                                   const std::vector<int>& validSubsetIndexes, const T* const peaksPtr,
                                   const int numberPeople, const unsigned int numberBodyParts,
                                   const unsigned int numberBodyPartPairs)
    {
        try
        {
            if (numberPeople > 0)
            {
                // Initialized to 0 for non-found keypoints in people
                poseKeypoints.reset({numberPeople, (int)numberBodyParts, 3}, 0);
                poseScores.reset(numberPeople);
            }
            else
            {
                poseKeypoints.reset();
                poseScores.reset();
            }
            const auto numberBodyPartsAndPAFs = numberBodyParts + numberBodyPartPairs;
            for (auto person = 0u ; person < validSubsetIndexes.size() ; person++)
            {
                const auto& personPair = peopleVector[validSubsetIndexes[person]];
                const auto& personVector = personPair.first;
                for (auto bodyPart = 0u; bodyPart < numberBodyParts; bodyPart++)
                {
                    const auto baseOffset = (person*numberBodyParts + bodyPart) * 3;
                    const auto bodyPartIndex = personVector[bodyPart];
                    if (bodyPartIndex > 0)
                    {
                        poseKeypoints[baseOffset] = peaksPtr[bodyPartIndex-2] * scaleFactor;
                        poseKeypoints[baseOffset + 1] = peaksPtr[bodyPartIndex-1] * scaleFactor;
                        poseKeypoints[baseOffset + 2] = peaksPtr[bodyPartIndex];
                    }
                }
                poseScores[person] = personPair.second / T(numberBodyPartsAndPAFs);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template <typename T>
    void connectBodyPartsCpu(Array<T>& poseKeypoints, Array<T>& poseScores, const T* const heatMapPtr,
                             const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
                             const int maxPeaks, const T interMinAboveThreshold, const T interThreshold,
                             const int minSubsetCnt, const T minSubsetScore, const T scaleFactor,
                             const bool maximizePositives)
    {
        try
        {
            // Parts Connection
            const auto& bodyPartPairs = getPosePartPairs(poseModel);
            const auto numberBodyParts = getPoseNumberBodyParts(poseModel);
            const auto numberBodyPartPairs = (unsigned int)(bodyPartPairs.size() / 2);
            if (numberBodyParts == 0)
                error("Invalid value of numberBodyParts, it must be positive, not " + std::to_string(numberBodyParts),
                      __LINE__, __FUNCTION__, __FILE__);

            // std::vector<std::pair<std::vector<int>, double>> refers to:
            //     - std::vector<int>: [body parts locations, #body parts found]
            //     - double: person subset score
            const auto peopleVector = createPeopleVector(
                heatMapPtr, peaksPtr, poseModel, heatMapSize, maxPeaks, interThreshold, interMinAboveThreshold,
                bodyPartPairs, numberBodyParts, numberBodyPartPairs);

            // Delete people below the following thresholds:
                // a) minSubsetCnt: removed if less than minSubsetCnt body parts
                // b) minSubsetScore: removed if global score smaller than this
                // c) maxPeaks (POSE_MAX_PEOPLE): keep first maxPeaks people above thresholds
            int numberPeople;
            std::vector<int> validSubsetIndexes;
            validSubsetIndexes.reserve(fastMin((size_t)maxPeaks, peopleVector.size()));
            removePeopleBelowThresholds(
                validSubsetIndexes, numberPeople, peopleVector, numberBodyParts, minSubsetCnt, minSubsetScore,
                maxPeaks, maximizePositives);

            // Fill and return poseKeypoints
            peopleVectorToPeopleArray(poseKeypoints, poseScores, scaleFactor, peopleVector, validSubsetIndexes,
                                      peaksPtr, numberPeople, numberBodyParts, numberBodyPartPairs);

            // Experimental code
            if (poseModel == PoseModel::BODY_25D)
                error("BODY_25D is an experimental branch which is not usable.", __LINE__, __FUNCTION__, __FILE__);
//                 connectDistanceMultiStar(poseKeypoints, poseScores, heatMapPtr, peaksPtr, poseModel, heatMapSize,
//                                          maxPeaks, scaleFactor, numberBodyParts, bodyPartPairs.size());
//                 connectDistanceStar(poseKeypoints, poseScores, heatMapPtr, peaksPtr, poseModel, heatMapSize,
//                                     maxPeaks, scaleFactor, numberBodyParts, bodyPartPairs.size());
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template OP_API void connectBodyPartsCpu(
        Array<float>& poseKeypoints, Array<float>& poseScores, const float* const heatMapPtr,
        const float* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks,
        const float interMinAboveThreshold, const float interThreshold, const int minSubsetCnt,
        const float minSubsetScore, const float scaleFactor, const bool maximizePositives);
    template OP_API void connectBodyPartsCpu(
        Array<double>& poseKeypoints, Array<double>& poseScores, const double* const heatMapPtr,
        const double* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks,
        const double interMinAboveThreshold, const double interThreshold, const int minSubsetCnt,
        const double minSubsetScore, const double scaleFactor, const bool maximizePositives);

    template OP_API std::vector<std::pair<std::vector<int>, float>> createPeopleVector(
        const float* const heatMapPtr, const float* const peaksPtr, const PoseModel poseModel,
        const Point<int>& heatMapSize, const int maxPeaks, const float interThreshold,
        const float interMinAboveThreshold, const std::vector<unsigned int>& bodyPartPairs,
        const unsigned int numberBodyParts, const unsigned int numberBodyPartPairs,
        const Array<float>& precomputedPAFs);
    template OP_API std::vector<std::pair<std::vector<int>, double>> createPeopleVector(
        const double* const heatMapPtr, const double* const peaksPtr, const PoseModel poseModel,
        const Point<int>& heatMapSize, const int maxPeaks, const double interThreshold,
        const double interMinAboveThreshold, const std::vector<unsigned int>& bodyPartPairs,
        const unsigned int numberBodyParts, const unsigned int numberBodyPartPairs,
        const Array<double>& precomputedPAFs);

    template OP_API void removePeopleBelowThresholds(
        std::vector<int>& validSubsetIndexes, int& numberPeople,
        const std::vector<std::pair<std::vector<int>, float>>& peopleVector,
        const unsigned int numberBodyParts,
        const int minSubsetCnt, const float minSubsetScore, const int maxPeaks, const bool maximizePositives);
    template OP_API void removePeopleBelowThresholds(
        std::vector<int>& validSubsetIndexes, int& numberPeople,
        const std::vector<std::pair<std::vector<int>, double>>& peopleVector,
        const unsigned int numberBodyParts,
        const int minSubsetCnt, const double minSubsetScore, const int maxPeaks, const bool maximizePositives);

    template OP_API void peopleVectorToPeopleArray(
        Array<float>& poseKeypoints, Array<float>& poseScores, const float scaleFactor,
        const std::vector<std::pair<std::vector<int>, float>>& peopleVector,
        const std::vector<int>& validSubsetIndexes, const float* const peaksPtr,
        const int numberPeople, const unsigned int numberBodyParts,
        const unsigned int numberBodyPartPairs);
    template OP_API void peopleVectorToPeopleArray(
        Array<double>& poseKeypoints, Array<double>& poseScores, const double scaleFactor,
        const std::vector<std::pair<std::vector<int>, double>>& peopleVector,
        const std::vector<int>& validSubsetIndexes, const double* const peaksPtr,
        const int numberPeople, const unsigned int numberBodyParts,
        const unsigned int numberBodyPartPairs);
}
