import logging
from typing import Set, List
import distance
import numpy as np
from sklearn.cluster import AffinityPropagation
import time
import os
import warnings
import logging

from evogfuzz.input import Input

warnings.filterwarnings("ignore", message="ConvergenceWarning: Affinity propagation did not converge, this model may return degenerate cluster centers and labels.")
warnings.filterwarnings("ignore", message="UserWarning: All samples have mutually equal similarities. Returning arbitrary cluster center(s).")


logger = logging.getLogger()


class Tournament:
    def __init__(
        self,
        test_inputs: Set[Input],
        tournament_rounds: int = 10,
        tournament_size: int = 10,
    ):
        self.test_inputs: Set[Input] = test_inputs
        self.tournament_rounds: int = tournament_rounds
        self.tournament_size: int = tournament_size

    def select_fittest_individuals(self):
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
                    cluster_str = ", ".join(cluster)
                    #logging.debug(" - *%s:* %s" % (exemplar, cluster_str))

                    exemplar_input_obj = [i for i in self.test_inputs if str(i)==exemplar][0]
                    if exemplar_input_obj.fitness == 1:
                        fittest.add(exemplar_input_obj)
                        if len(cluster)>1:
                            for sample_index, sample in enumerate(cluster):
                                sample_input_obj = [i for i in self.test_inputs if str(i)==sample][0]
                                if sample_input_obj.fitness == 1:
                                    fittest.add(sample_input_obj)
                                if sample_index>=(aimed_sample_count/len(cluster_list))+1 or len(fittest)>aimed_sample_count:
                                    break
                            if len(fittest)>aimed_sample_count:
                                    break
                    if index>=max_variety and len(fittest)>=max_variety:
                        break
                    if max_distance>initial_max_distance and index>=len(cluster_list)/2 and len(fittest)==0:
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
