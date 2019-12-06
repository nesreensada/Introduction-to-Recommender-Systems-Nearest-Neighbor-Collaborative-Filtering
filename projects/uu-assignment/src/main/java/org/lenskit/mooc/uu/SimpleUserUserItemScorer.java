package org.lenskit.mooc.uu;

import com.google.common.collect.Maps;
import com.google.common.math.LongMath;
import it.unimi.dsi.fastutil.longs.Long2DoubleMap;
import it.unimi.dsi.fastutil.longs.Long2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.longs.Long2DoubleSortedMap;
import org.lenskit.api.Result;
import org.lenskit.api.ResultMap;
import org.lenskit.basic.AbstractItemScorer;
import org.lenskit.data.dao.DataAccessObject;
import org.lenskit.data.entities.CommonAttributes;
import org.lenskit.data.ratings.Rating;
import org.lenskit.results.Results;
import org.lenskit.util.ScoredIdAccumulator;
import org.lenskit.util.TopNScoredIdAccumulator;
import org.lenskit.util.collections.LongUtils;
import org.lenskit.util.io.ObjectStream;
import org.lenskit.util.math.Scalars;
import org.lenskit.util.math.Vectors;

import javax.annotation.Nonnull;
import javax.inject.Inject;
import java.util.*;

/**
 * User-user item scorer.
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class SimpleUserUserItemScorer extends AbstractItemScorer {
    private final DataAccessObject dao;
    private final int neighborhoodSize;

    /**
     * Instantiate a new user-user item scorer.
     * @param dao The data access object.
     */
    @Inject
    public SimpleUserUserItemScorer(DataAccessObject dao) {
        this.dao = dao;
        neighborhoodSize = 30;
    }

    @Nonnull
    @Override
    public ResultMap scoreWithDetails(long user, @Nonnull Collection<Long> items) {
        Long2DoubleOpenHashMap userRatingVector = getUserRatingVector(user);
        Double sum = new Double(0);
        Double mean = new Double(0);

        List<Result> results = new ArrayList<>();

        ObjectStream<Rating> ratings = dao.query(Rating.class).stream();
        Map<Long, Map<Long, Double>> itemMap = new HashMap<>();
        // get the users ratings for each item and construct map for the raters
        for(Rating rating: ratings){
            Map<Long, Double> newUserVector = new HashMap<>();
            Long key = rating.getItemId();

            if(itemMap.containsKey(key)){
                Map<Long, Double> userVector = itemMap.get(key);
                userVector.put(rating.getUserId(), rating.getValue());

                itemMap.put(key,userVector);
            }else{
                newUserVector.put(rating.getUserId(), rating.getValue());
                itemMap.put(key, newUserVector);
            }
        }
        //predict rating of an item
        for (Long item:items){
            Map<Long, Double> users =  itemMap.get(item);
            Map<Long, Map<Long, Double>> usersValue = new HashMap<>();

            for (Map.Entry<Long, Double> entry: users.entrySet()){
                Long uId = entry.getKey();
                if(uId != user){
                    usersValue.put(uId, getUserRatingVector(uId));
                }
            }
            // get n 30 neighbours
            Map<Long, Double> neighbours = calculate30TopNeighbours(userRatingVector, usersValue);
            if (neighbours.size()>2){ // score for more than 2
                Double num = new Double(0);
                Double dem = new Double(0);

                Map<Long, Double> meanVal =  calculateMeanRating(user, userRatingVector, usersValue);

                for(Map.Entry<Long, Double> e2: neighbours.entrySet()){
                    Long key = e2.getKey();
                    Double similarityVal = e2.getValue();

                    num += (usersValue.get(key).get(item) - meanVal.get(key)) * similarityVal;
                    dem += similarityVal;
                }
                Double predicated =  meanVal.get(user) + (num/ dem);
                results.add(Results.create(item, predicated));
            }
        }

        return Results.newResultMap(results);

    }

    /**
     * Get a user's rating vector.
     * @param user The user ID.
     * @return The rating vector, mapping item IDs to the user's rating
     *         for that item.
     */
    private Long2DoubleOpenHashMap getUserRatingVector(long user) {
        List<Rating> history = dao.query(Rating.class)
                                  .withAttribute(CommonAttributes.USER_ID, user)
                                  .get();

        Long2DoubleOpenHashMap ratings = new Long2DoubleOpenHashMap();
        for (Rating r: history) {
            ratings.put(r.getItemId(), r.getValue());
        }

        return ratings;
    }
    private Map<Long, Double> calculateMeanRating(long user,
                                                  Map<Long, Double>userRatingVector, Map<Long, Map<Long, Double>> userVector) {
        Map<Long, Double> meanRating = new HashMap<>();
        for (Map.Entry<Long, Map<Long, Double>> entry: userVector.entrySet()){
            Long2DoubleMap uVector = LongUtils.frozenMap(entry.getValue());
            Double mean = Vectors.mean(uVector);
            meanRating.put(entry.getKey(), mean);
        }

        Long2DoubleMap targetVector = LongUtils.frozenMap(userRatingVector);
        Double targetMean = Vectors.mean(targetVector);
        meanRating.put(user, targetMean);
        return meanRating;
    }
    private Map<Long, Double> calculate30TopNeighbours(Map<Long, Double> userRatingVector,
                                                       Map<Long, Map<Long, Double>> userVector) {

        List<Result> results =  new ArrayList<>();
        // get the similarity between users
        for (Map.Entry<Long, Map<Long, Double>> entry: userVector.entrySet()){
            Long uId = entry.getKey();
            Double similarity = calculateSimilarity(userRatingVector,
                    entry.getValue());
            results.add(Results.create(uId, similarity));
        }

        Collections.sort(results, new Comparator<Result>() {
            @Override
            public int compare(Result result, Result t1) {
                return result.getScore() > t1.getScore()?-1:(result.getScore()<t1.getScore())?1:0;
            }
        });
        Map<Long, Double> top30 = new HashMap<>();
        int count = 0;
        while(count<30 && count <results.size()){
            Result r = results.get(count);
            Long nb_id = r.getId();
            Double similarityVal = r.getScore();
            if (similarityVal>0){
                top30.put(nb_id, similarityVal);
            }
            count++;
        }
        return top30;
    }
    private Double calculateSimilarity(Map<Long, Double> targetUserRatingMap,
                                       Map<Long, Double> userRatingMap) {
        Map<Long, Double> targetMap = new HashMap<Long, Double>(targetUserRatingMap);
        Map<Long, Double> userMap = new HashMap<Long, Double>(userRatingMap);

        Long2DoubleMap m1 = LongUtils.frozenMap(targetMap);
        Long2DoubleMap m2 = LongUtils.frozenMap(userMap);

        Double targetSq = new Double(0);
        Double userSq =  new Double(0);
        Double targetMean = new Double(Vectors.mean(m1));
        Double userMean = new Double(Vectors.mean(m2));

        // target user values
        for(Map.Entry<Long, Double> entry: targetMap.entrySet()){
            Double newVal = entry.getValue() - targetMean;
            entry.setValue(newVal);
            targetSq += newVal * newVal;
        }
        //  user values
        for(Map.Entry<Long, Double> entry: userMap.entrySet()){
            Double newVal = entry.getValue() - userMean;
            entry.setValue(newVal);
            userSq += newVal * newVal;
        }
        Long2DoubleMap finalM1 = LongUtils.frozenMap(targetMap);
        Long2DoubleMap finalM2 = LongUtils.frozenMap(userMap);

        Double denominator = Math.sqrt(targetSq) * Math.sqrt(userSq);
        Double similarityVal = new Double(Vectors.dotProduct(finalM1, finalM2)/ denominator);

        return similarityVal;

    }

}
