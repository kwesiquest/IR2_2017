from os.path import isfile, split
import pickle
import gzip
from stopwords import stopwords
from nGramTokenMap import tokenise
from collections import Iterable, defaultdict


def save_query_object_to_pickle(query_object):
    filename = query_object.get_query_object_file_name()
    print('Pickling Query Object as {}'.format(filename))
    with open(filename, 'wb') as handle:
        pickle.dump(query_object, handle, pickle.HIGHEST_PROTOCOL)
    print('Pickled!')
    return None


class Query(object):

    def __init__(self, path, stopwords=stopwords, entity_ids=None):
        self.path = path
        self.main_category = self.get_main_category
        self.stopwords = stopwords
        self.entity_ids = entity_ids
        if self.pickle_exists():
            self.load_queries_from_pickle(self, self.get_query_object_file_name())
        else:
            self.queries = self.get_queries

    @property
    def get_main_category(self):
        filename = split(self.path)[1]
        start, end = 'meta_', '.json.gz'
        assert filename.startswith(start) and filename.endswith(end)
        main_category = filename[len(start): -len(end)]
        main_category = main_category.replace("_", " ")
        main_category = main_category.replace("and", "&")
        print(main_category)
        return main_category

    @property
    def get_queries(self):
        query_dict = defaultdict(list)
        for metadata in self.parse():
            entity_id = metadata['asin']
            if entity_id in self.entity_ids:
                categories = metadata['categories']
                filtered_categories = [cats for cats in categories if cats[0] == self.main_category]
                query_categories = [sections[1:] for sections in filtered_categories]
                for cats in query_categories:
                    query = tokenise(' '.join(set(word for s in (cat.split() for cat in cats) for word in s)))
                    query = ' '.join(query)

                    query_dict[query].append(entity_id)
        return query_dict

    # Reference: https://stackoverflow.com/a/2158532/5960561
    def flatten(self, categories):
        for cat in categories:
            if isinstance(cat, Iterable) and not isinstance(cat, (str, bytes)):
                yield from self.flatten(cat)
            else:
                yield tokenise(cat)

    # Reference: http://jmcauley.ucsd.edu/data/amazon/links.html
    def parse(self):
        g = gzip.open(self.path, 'rb')
        for l in g:
            yield eval(l)

    def get_query_object_file_name(self):
        path_to, filename = split(self.path)
        start = 'meta'
        end = '.json.gz'
        assert filename.startswith(start) and filename.endswith(end)
        subset = filename[len(start): -len(end)]
        picklename = 'Query_object{}.pickle'.format(subset)
        if not len(path_to):
            return picklename
        else:
            return path_to + '/' + picklename

    def pickle_exists(self):
        filename = self.get_query_object_file_name()
        return isfile(filename)

    @staticmethod
    def load_queries_from_pickle(queries_instance, pickle_file):
        print("Unpickling:", pickle_file)
        with open(pickle_file, 'rb') as handle:
            queries_object = pickle.load(handle)
        queries_instance.queries = queries_object.queries
        return queries_object

if __name__ == '__main__':
    path = 'meta_Home_and_Kitchen.json.gz'
    chosen_entities = ['0615391206', '0689027818', '0912696591', '1223070743', '1567120709', '1891747401', '1983475912', '2042037265', '7213035835', '7502151168', '7535549756', '7802215811', '9177124405', '9178894395', 'B00000DMDJ', 'B00000JGRP', 'B00000JGRQ', 'B00000JGRS', 'B00000JGRT', 'B0000223YC', 'B00002255K', 'B00002N5FO', 'B00002N5Z9', 'B00002N5ZB', 'B00002N5ZP', 'B00002N5ZQ', 'B00002N601', 'B00002N602', 'B00002N62X', 'B00002N630', 'B00002N636', 'B00002N63A', 'B00002N6SQ', 'B00002N86C', 'B00002N8CX', 'B00002N8CZ', 'B00002N8DF', 'B00002NAPD', 'B00002NB5I', 'B00002NBSO', 'B00002NC6E', 'B00002NC6F', 'B00002ND67', 'B00002ND6A', 'B00004OCIP', 'B00004OCIQ', 'B00004OCIR', 'B00004OCIS', 'B00004OCIU', 'B00004OCIV', 'B00004OCIW', 'B00004OCIX', 'B00004OCIY', 'B00004OCIZ', 'B00004OCJ2', 'B00004OCJ6', 'B00004OCJ7', 'B00004OCJ9', 'B00004OCJG', 'B00004OCJJ', 'B00004OCJK', 'B00004OCJM', 'B00004OCJN', 'B00004OCJO', 'B00004OCJQ', 'B00004OCJS', 'B00004OCJU', 'B00004OCJW', 'B00004OCJY', 'B00004OCK0', 'B00004OCK3', 'B00004OCK5', 'B00004OCKG', 'B00004OCKK', 'B00004OCKO', 'B00004OCKR', 'B00004OCKT', 'B00004OCKU', 'B00004OCKX', 'B00004OCL2', 'B00004OCL8', 'B00004OCL9', 'B00004OCLA', 'B00004OCLC', 'B00004OCLE', 'B00004OCLF', 'B00004OCLH', 'B00004OCLK', 'B00004OCLW', 'B00004OCM4', 'B00004OCM5', 'B00004OCM9', 'B00004OCMB', 'B00004OCME', 'B00004OCMH', 'B00004OCMJ', 'B00004OCMM', 'B00004OCMN', 'B00004OCMO', 'B00004OCMP']
    queries = Query(path, entity_ids=chosen_entities)
    save_query_object_to_pickle(queries)
    single_entity = 0
    many_entities = 0
    for query, entity_id_list in queries.queries.items():
        if len(entity_id_list) > 1:
            many_entities += 1
            print(query, entity_id_list)
        else:
            single_entity += 1
    print("Queries with one entity: {}".format(single_entity))
    print("Queries with many entities: {}".format(many_entities))

