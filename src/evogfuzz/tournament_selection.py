import logging
from typing import Set, List
import distance
from sklearn.cluster import AffinityPropagation
import time
import logging
from alhazen import feature_collector
from evogfuzz.helper import Tournament_Selection_Mode
from fuzzingbook.GrammarFuzzer import Grammar
from evogfuzz.input import Input
import pandas as pd
from sklearn.metrics import pairwise_distances
import fastcluster
from scipy.cluster.hierarchy import fcluster
import numpy as np
from scipy.stats import median_abs_deviation
from statistics import median
import Levenshtein
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from jarowinkler import *
from scipy.spatial.distance import cdist

def levenshtein_ratio(inp1, inp2):
    return 1-Levenshtein.ratio(str(inp1), str(inp2))
def jaro_jaro_winkler_metric(inp1,inp2):
    return 1-jarowinkler_similarity(str(inp1), str(inp2))
class Tournament:
    def __init__(
            self,
            test_inputs: Set[Input],
            tournament_rounds: int = 10,
            tournament_size: int = 10,
            grammar: Grammar = None,
            tournament_selection_mode: Tournament_Selection_Mode = Tournament_Selection_Mode.NORMAL,
            feature_vectors_dataframe: pd.DataFrame = None
    ):
        self.test_inputs: Set[Input] = test_inputs
        self.tournament_rounds: int = tournament_rounds
        self.tournament_size: int = tournament_size
        self.grammar = grammar
        self.tournament_selection_mode = tournament_selection_mode
        self.distance_matrix = None
        self.index = None
    def select_fittest_individuals(self):
        if self.tournament_selection_mode == Tournament_Selection_Mode.NORMAL:
            return self.select_fittest_individuals_normal()

        elif self.tournament_selection_mode == Tournament_Selection_Mode.HIERARCHICAL_FEATURE_COS:
            return self.select_fittest_individuals_hierarchical_feature_cos()

        elif self.tournament_selection_mode == Tournament_Selection_Mode.HIERARCHICAL_LEVENSHTEIN:
            return self.select_fittest_individuals_hierarchical_levenshtein()

        elif self.tournament_selection_mode == Tournament_Selection_Mode.HIERARCHICAL_JARO:
            return self.select_fittest_individuals_hierarchical_jaro()

        else:
            print("unknown enum, NORMAL mode selected")
            return self.select_fittest_individuals_normal()

    def select_fittest_individuals_normal(self):
        # print(len(self.test_inputs), self.tournament_rounds)
        # assert self.tournament_rounds < len(self.test_inputs)

        fittest: Set[Input] = set()

        try:
            for _ in range(self.tournament_rounds):
                current_round: List[Input] = list(self.test_inputs)[
                                             : self.tournament_size
                                             ]
                for inp in current_round:
                    self.test_inputs.remove(inp)
                fi = sorted(
                    current_round, key=lambda inp: inp.fitness, reverse=False
                ).pop()
                fittest.add(fi)
        except IndexError:
            logging.debug("Tournament Size too big! No more Inputs left to select!")

        return fittest

    def select_fittest_individuals_hierarchical_levenshtein(self):
        self.calc_dist_matrix(levenshtein_ratio)
        return self.select_fittest_individuals_hierarchical_custom_method()

    def calc_dist_matrix(self, method):
        arr1 = np.array(list(self.test_inputs))
        matrix = cdist(arr1.reshape(-1, 1), arr1.reshape(-1, 1), lambda x, y: method(x[0], y[0]))
        self.index = arr1
        self.distance_matrix = pd.DataFrame(data=matrix, index=arr1, columns=arr1)

    def select_fittest_individuals_hierarchical_custom_method(self):
        clusters_sets = self.distance_matrix_2_clusters_sets(self.distance_matrix)
        upper_ones = self.filter_clusters_by_bug_log_precentile_median_plus_mad(clusters_sets)
        self.test_inputs = set([item for sublist in upper_ones for item in sublist])
        #self.test_inputs = set([x for x in self.test_inputs if len(str(x)) < 50]) # had 2 be done
        return self.select_fittest_individuals_normal()

    def select_fittest_individuals_hierarchical_jaro(self):
        self.calc_dist_matrix(jaro_jaro_winkler_metric)
        return self.select_fittest_individuals_hierarchical_custom_method()

    def select_fittest_individuals_hierarchical_feature_cos(self):
        feature_vectors_dataframe = self.input_2_dataframe_with_features()
        self.distance_matrix = pairwise_distances(feature_vectors_dataframe, metric="cosine")
        return self.select_fittest_individuals_hierarchical_custom_method()

    @staticmethod
    def filter_clusters_by_bug_log_precentile_median_plus_mad(clusters_sets):
        def get_bug_log_precentile(clusters_sets):
            from math import log
            clusters_2_log_perc = {}
            for elem in clusters_sets:
                fittnes_of_cluster = [x.fitness for x in clusters_sets[elem]]
                log_fittnes_prec = log(sum(fittnes_of_cluster)+1)/log(len(fittnes_of_cluster)+1)
                clusters_2_log_perc[elem]=log_fittnes_prec
            return clusters_2_log_perc

        clusters_log_perc = get_bug_log_precentile(clusters_sets)
        numbers = [clusters_log_perc[x] for x in clusters_sets]
        median_value = median(numbers)
        median_absolute_deviation = median_abs_deviation(numbers)
        #median_minus_mad = median_value - median_absolute_deviation
        median_plus_mad = median_value + median_absolute_deviation
        #middle_ones = [clusters_sets[x] for x in clusters_sets if
        #               median_minus_mad < clusters_log_perc[x] < median_plus_mad]
        upper_ones = [clusters_sets[x] for x in clusters_sets if
                       median_plus_mad < clusters_log_perc[x]]
        return upper_ones

    def input_2_dataframe_with_features(self):
        def inputs_to_feature_vectors_func():
            inputs_to_feature_vectors = {}
            collector = feature_collector.Collector(self.grammar)
            feature_name_2_key = {}
            for elem in collector.get_all_features():
                feature_name_2_key[elem.name] = elem.name + " " + elem.key
            for sample in self.test_inputs:
                gen_features = collector.collect_features(
                    Input(tree=sample.tree
                          )
                )

                gen_features2 = {}
                for elem in gen_features:
                    gen_features2[feature_name_2_key[elem]] = gen_features[elem]

                inputs_to_feature_vectors[sample] = gen_features2
            return inputs_to_feature_vectors

        feature_vectors_dataframe = pd.DataFrame.from_dict(inputs_to_feature_vectors_func()).T
        self.index = feature_vectors_dataframe.index
        return feature_vectors_dataframe

    def select_fittest_individuals_julius(self):
        # logging.debug(len(self.test_inputs), self.tournament_rounds)
        # assert self.tournament_rounds < len(self.test_inputs)

        fittest: Set[Input] = set()


        max_variety = 15
        aimed_sample_count = 30
        initial_max_distance = 40
        max_distance = initial_max_distance

        for i in list(self.test_inputs):
            if len(str(i))>max_distance:
                max_distance = len(str(i))

        inputs_to_feature_vectors={}
        collector = feature_collector.Collector(self.grammar)
        for sample in self.test_inputs:
            gen_features = collector.collect_features(
                Input(tree=sample.tree
                      )
            )
            inputs_to_feature_vectors[sample] = gen_features

        input_list = np.asarray([str(i) for i in list(self.test_inputs)])

        if max_variety>=len(input_list):
            max_variety = int(len(input_list)/2)+1
        logging.debug("input sample count: "+str(len(input_list)))

        # if max_distance was not increased, use clustering for selection
        if max_distance <= initial_max_distance:
            #input_list = input_list[0:max(2,int(150/max_distance))]
            #logging.debug("corrected input sample count because of long strings: "+str(len(input_list)))

            try:
                start = time.time()
                lev_similarity = float(-1)*np.array([[distance.levenshtein(i1,i2) for i1 in input_list if distance.levenshtein(i1,i2)<=max_distance] for i2 in input_list])
                lev_calc_time = time.time()-start
                if lev_calc_time>0.25:
                    logging.debug("long levenshtein calculation time: "+str(lev_calc_time)+"s")


                affprop = AffinityPropagation(affinity="precomputed", damping=0.7)
                affprop.fit(lev_similarity)

                cluster_list = np.unique(affprop.labels_)

                for index, cluster_id in enumerate(cluster_list):
                    exemplar = input_list[affprop.cluster_centers_indices_[cluster_id]]
                    cluster = np.unique(input_list[np.nonzero(affprop.labels_==cluster_id)])
                    #cluster_str = ", ".join(cluster)
                    #logging.debug(" - *%s:* %s" % (exemplar, cluster_str))

                    exemplar_input_obj = [i for i in self.test_inputs if str(i) == exemplar][0]
                    if exemplar_input_obj.fitness == 1:
                        fittest.add(exemplar_input_obj)
                        if len(cluster) > 1:
                            for sample_index, sample in enumerate(cluster):
                                sample_input_obj = [i for i in self.test_inputs if str(i) == sample][0]
                                if sample_input_obj.fitness == 1:
                                    fittest.add(sample_input_obj)
                                if sample_index >= (aimed_sample_count/len(cluster_list))+1 or len(fittest) > aimed_sample_count:
                                    break
                            if len(fittest) > aimed_sample_count:
                                    break
                    if index >= max_variety and len(fittest) >= max_variety:
                        break
                    if max_distance > initial_max_distance and index >= len(cluster_list)/2 and len(fittest) == 0:
                        break

            except ValueError:
                pass

        # for long string samples, use primitive selection
        else:

            logging.debug("og")
            try:
                for _ in range(self.tournament_rounds):
                    current_round: List[Input] = list(self.test_inputs)[
                        : self.tournament_size
                    ]
                    for inp in current_round:
                        self.test_inputs.remove(inp)
                    fi = sorted(
                        current_round, key=lambda inp: inp.fitness, reverse=False
                    ).pop()
                    fittest.add(fi)
            except IndexError:
                logging.debug("Tournament Size too big! No more Inputs left to select!")

        logging.debug("added input samples: "+str(len(fittest)))
        
        if len(fittest)/len(input_list)<0.1 and len(input_list)<10:
            logging.debug(f"less than 10%, only {len(fittest)}/{len(input_list)} added inputs out of this this list:")
            for i in input_list:
                if [sample for sample in self.test_inputs if str(sample)==i][0] in fittest:
                    logging.debug("\033[91m", end="")
                else:
                    logging.debug("\033[0m", end="")
                logging.debug(str(i)+" ", end="")
                logging.debug("\033[0m", end="")
        if len(fittest)/len(input_list)>=0.9:
            logging.debug(f"more than 90%, {len(fittest)}/{len(input_list)} added inputs out of this this list:")
            for i in input_list:
                if [sample for sample in self.test_inputs if str(sample)==i][0] in fittest:
                    logging.debug("\033[96m", end="")
                else:
                    logging.debug("\033[0m", end="")
                logging.debug(str(i)+" ", end="")
                logging.debug("\033[0m", end="")
            

        return fittest

    def distance_matrix_2_clusters_sets(self, distance_matrix):
        dendogramm = fastcluster.linkage(distance_matrix, 'ward', preserve_input=False)

        num_clust = int(len(self.test_inputs) ** 0.5)

        feature_vectors_index = self.index
        clusters = fcluster(dendogramm, num_clust, criterion='maxclust')

        clusters_sets = {}

        for index in range(len(feature_vectors_index)):
            clusters_index = clusters[index]
            input_string = feature_vectors_index[index]
            if clusters_index not in clusters_sets:
                clusters_sets[clusters_index] = set()
            clusters_sets[clusters_index].add(feature_vectors_index[index])

        return clusters_sets