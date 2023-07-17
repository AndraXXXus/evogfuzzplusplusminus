import logging
import os
from typing import Set, List
import distance
from sklearn.cluster import AffinityPropagation
import time
import logging
from alhazen import feature_collector
from evogfuzz.helper import Tournament_Selection_Mode
from fuzzingbook.GrammarFuzzer import Grammar
from evogfuzz.input import Input
import jaro
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

RED = "\033[91m"
NORMAL = "\033[0m"

#print(os.environ.get("Tournament_Selection_Mode"))

def Levenshtein_ratio(inp1,inp2):
    return Levenshtein.ratio(str(inp1), str(inp2))
def jaro_jaro_winkler_metric(inp1,inp2):
    return jaro.jaro_winkler_metric(str(inp1), str(inp2))
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
        self.feature_vectors_dataframe = feature_vectors_dataframe

    def select_fittest_individuals(self):
        self.tournament_selection_mode = Tournament_Selection_Mode(int(os.environ.get("Tournament_Selection_Mode", 0)))
        fittest_individuals = None
        input_count = len(self.test_inputs)
        print(f"test_inputs[{input_count}]:")
        fit_count = 0
        for i in self.test_inputs:
            print(str(i)+" ", end="")
            if i.fitness > 0:
                fit_count+=1
        print(f"\nfit_count: {fit_count}")
        print()
        if self.tournament_selection_mode == Tournament_Selection_Mode.NORMAL:
            fittest_individuals =  self.select_fittest_individuals_normal()

        elif self.tournament_selection_mode == Tournament_Selection_Mode.HIERARCHICAL_FEATURE_COS:
            fittest_individuals =  self.select_fittest_individuals_hierarchical_feature_cos()

        elif self.tournament_selection_mode == Tournament_Selection_Mode.HIERARCHICAL_LEVENSHTEIN:
            #fittest_individuals =  self.select_fittest_individuals_hierarchical_levenshtein()
            fittest_individuals = self.select_fittest_individuals_julius()

        elif self.tournament_selection_mode == Tournament_Selection_Mode.HIERARCHICAL_JARO:
            fittest_individuals =  self.select_fittest_individuals_hierarchical_jaro()

        else:
            print("unknown enum, NORMAL mode selected")
            fittest_individuals =  self.select_fittest_individuals_normal()

        print(f"fittest[{len(fittest_individuals)}]:")
        
        correct_count = 0
        for fi in fittest_individuals:
            if fi.fitness == 0:
                print(RED, end="")
            else:
                correct_count +=1
                print("\033[0m", end="")
            print(str(fi)+" ", end="")
        if len(fittest_individuals)==0:
            print(RED+"[empty]")
        print("\033[0m\n\n---------------")

        if len(fittest_individuals)>0:
            correctness = round((correct_count/len(fittest_individuals))*100, 2)
        else:
            correctness = "null"

        file_name = ""

        if os.environ.get("Tournament_Selection_Mode")=="0":
            file_name = "original"
        elif os.environ.get("Tournament_Selection_Mode")=="2":
            file_name = "julius_levenshtein"

        with open(file_name+".csv","a") as f:
            f.write(str(input_count)+","+str(len(fittest_individuals))+","+str(correctness)+"\n")
        return fittest_individuals

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
        return self.select_fittest_individuals_hierarchical_custom_method(Levenshtein_ratio)

    def select_fittest_individuals_hierarchical_custom_method(self, method):
        self.feature_vectors_dataframe = self.calculate_cosine_similarity()
        distance_matrix = pairwise_distances(self.feature_vectors_dataframe, metric=method)
        clusters_sets = self.distance_matrix_2_clusters_sets(distance_matrix)
        middle_ones = self.filter_clusters_by_bug_log_precentile_median_plus_minus_mad(clusters_sets)
        self.test_inputs = set([item for sublist in middle_ones for item in sublist])
        return self.select_fittest_individuals_normal()

    def select_fittest_individuals_hierarchical_jaro(self):
        return self.select_fittest_individuals_hierarchical_custom_method(jaro_jaro_winkler_metric)

    def select_fittest_individuals_hierarchical_feature_cos(self):
        self.feature_vectors_dataframe = self.input_2_dataframe_with_features()
        return self.select_fittest_individuals_hierarchical_custom_method("cosine")

    @staticmethod
    def filter_clusters_by_bug_log_precentile_median_plus_minus_mad(clusters_sets):
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
        median_minus_mad = median_value - median_absolute_deviation
        median_plus_mad = median_value + median_absolute_deviation
        middle_ones = [clusters_sets[x] for x in clusters_sets if
                       median_minus_mad < clusters_log_perc[x] < median_plus_mad]
        return middle_ones

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

        feature_vectors_dataframe = pd.DataFrame.from_dict(inputs_to_feature_vectors_func()).T
        return feature_vectors_dataframe

    def select_fittest_individuals_julius(self):
        # logging.debug(len(self.test_inputs), self.tournament_rounds)
        # assert self.tournament_rounds < len(self.test_inputs)

        fittest: Set[Input] = set()


        max_variety = 15
        aimed_sample_count = 24
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

        # if max_distance was not increased, use clustering for selection
        if max_distance <= initial_max_distance:
            #input_list = input_list[0:max(2,int(150/max_distance))]
            #logging.debug("corrected input sample count because of long strings: "+str(len(input_list)))

            try:
                start = time.time()
                lev_similarity = float(-1)*np.array([[distance.levenshtein(i1,i2) for i1 in input_list if distance.levenshtein(i1,i2)<=max_distance] for i2 in input_list])
                lev_calc_time = time.time()-start
                if lev_calc_time>=1:
                    print("long levenshtein calculation time: "+str(lev_calc_time)+"s")


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
            print()
            print(RED+"OG")
            print(NORMAL)

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

    def calculate_cosine_similarity(self):
        ordered_inputs = [x for x in self.test_inputs]
        strings = [str(x) for x in ordered_inputs]
        vectorizer = CountVectorizer(analyzer='char')

        matrix = vectorizer.fit_transform(strings)

        similarity_matrix = cosine_similarity(matrix)

        df = pd.DataFrame(similarity_matrix, columns=ordered_inputs, index=ordered_inputs)

        return df

    def distance_matrix_2_clusters_sets(self, distance_matrix):
        dendogramm = fastcluster.linkage(distance_matrix, 'ward', preserve_input=False)

        num_clust = int(self.feature_vectors_dataframe.shape[0] ** 0.5)

        feature_vectors_index = self.feature_vectors_dataframe.index
        clusters = fcluster(dendogramm, num_clust, criterion='maxclust')

        clusters_sets = {}

        for index in range(len(feature_vectors_index)):
            clusters_index = clusters[index]
            input_string = feature_vectors_index[index]
            if clusters_index not in clusters_sets:
                clusters_sets[clusters_index] = set()
            clusters_sets[clusters_index].add(feature_vectors_index[index])

        return clusters_sets