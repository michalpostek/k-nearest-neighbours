namespace KNearestNeighbours;

public static class KNearestNeighbours<TLabel> where TLabel : struct, Enum 
{
    public static TLabel? Classify(double[] obj, List<Tuple<double[], TLabel>> trainingData, int k, Func<double[], double[], double> distanceMetric) 
    {
        if (k < 1)
        {
            throw new ArgumentException("K must be greater than zero");
        }

        var trainingDataByLabel = GroupTrainingData(trainingData);

        if (trainingDataByLabel.Keys.Any(label => trainingDataByLabel[label].Length < k))
        {
            throw new ArgumentException("The training data does not contain enough neighbours for at least one class");
        }
        
        var resultByLabel = trainingDataByLabel.ToDictionary((kvp) => kvp.Key, kvp => SumKNearestNeighbourDistances(obj, kvp.Value, distanceMetric, k));

        return TryGetUniqueMinKey(resultByLabel);
    }

    private static double SumKNearestNeighbourDistances(double[] obj, double[][] neighbours, Func<double[], double[], double> distanceMetric, int k)
    {
        return neighbours.Select(neighbour => distanceMetric(neighbour, obj)).OrderBy(distance => distance).Take(k).Sum();
    }

    private static Dictionary<TLabel, double[][]> GroupTrainingData(List<Tuple<double[], TLabel>> trainingData)
    {
        return trainingData
            .GroupBy(sample => sample.Item2)
            .ToDictionary(group => group.Key, group => group.Select(sample => sample.Item1).ToArray());
    } 
    
    private static TLabel? TryGetUniqueMinKey(Dictionary<TLabel, double> dict)
    {
        if (dict.Count == 0)
        {
            return null;
        }

        var min = dict.Min(kv => kv.Value);
        var keysWithMin = dict.Where(kv => AreCloseEnough(kv.Value, min)).Select(kv => kv.Key).ToList();

        if (keysWithMin.Count != 1)
        {
            return null;
        }

        return keysWithMin.First();
    }

    private static bool AreCloseEnough(double a, double b, double epsilon = 1e-9)
    { 
        return Math.Abs(a - b) < epsilon;
    }
}
