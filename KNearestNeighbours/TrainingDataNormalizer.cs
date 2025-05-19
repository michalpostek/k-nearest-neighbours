namespace KNearestNeighbours;

public static class TrainingDataNormalizer
{
    public static List<Tuple<double[], TClass>> Normalize<TClass>(List<Tuple<double[], TClass>> trainingData)
    {
        var (min, max) = GetMinMax(trainingData);

        return trainingData.Select(sample => new Tuple<double[], TClass>(
            sample.Item1.Select(value => NormalizeSingleValue(value, min, max)).ToArray(),
            sample.Item2
        )).ToList();
    }

    private static double NormalizeSingleValue(double value, double min, double max)
    {
        return (value - min) / (max - min);
    }

    private static (double Min, double Max) GetMinMax<TClass>(List<Tuple<double[], TClass>> trainingData)
    {
        if (trainingData.Count == 0)
        {
            throw new ArgumentException("No training data");
        }

        var allValues = trainingData.SelectMany(x => x.Item1).ToArray();

        return (
            allValues.Min(),
            allValues.Max()
        );
    }
}
