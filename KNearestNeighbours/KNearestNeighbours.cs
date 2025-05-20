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
        var resultByLabel = new List<(TLabel, double)>();
        
        foreach (var (label, samples) in trainingDataByLabel)
        {
            if (trainingDataByLabel[label].Length < k)
            {
                throw new ArgumentException($"The training data does not contain at least {k} samples of type {label}");
            }

            var result = samples.Select(sample => distanceMetric(sample, obj)).OrderBy(distance => distance).Take(k).Sum();
            resultByLabel.Add((label, result));
        }
        
        resultByLabel.Sort((a, b) => a.Item2.CompareTo(b.Item2));

        if (resultByLabel[0].Item2.Equals(resultByLabel[1].Item2))
        {
            return null;
        }

        return resultByLabel[0].Item1;
    }

    private static Dictionary<TLabel, double[][]> GroupTrainingData(List<Tuple<double[], TLabel>> trainingData)
    {
        return trainingData
            .GroupBy(sample => sample.Item2)
            .ToDictionary(group => group.Key, group => group.Select(sample => sample.Item1).ToArray());
    }
}
