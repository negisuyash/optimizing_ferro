import re, math, inspect, os, logging.config
import googlemaps
import pandas as pd
import timeit
import threading


logger = logging.getLogger(__name__)


class BagPacker:
    df=None

    def __init__(self, pkgs):
        try:
            logger.info(pkgs)
            self.df = pd.DataFrame(list(zip([pkg['address'] for pkg in pkgs], [pkg['ID'] for pkg in pkgs],
                                       [pkg['shipment_ID'] for pkg in pkgs], [pkg['tag_ID'] for pkg in pkgs])),
                              columns=['addr', 'pkg_ID','shipment_ID','tag_ID']).dropna()

            #df.to_csv(os.getenv('DATASETS_FOLDER')+'pairing.csv')
        except Exception as e:
            logger.error("Issue with " + str(inspect.stack()[0][3]), exc_info=True)

    def prepare_bags(self, pkgs, params):
        #df = pd.read_csv(os.getenv('DATASETS_FOLDER') + 'pairing.csv',
                         #usecols=['addr', 'pkg_ID', 'shipment_ID', 'tag_ID']).dropna()
        df=self.df
        df['located'] = [0 for i in range(len(df))]
        df['breaked_addr'] = BagPacker.clean_address(df['addr'])
        del df['addr']
        #df.to_csv(os.getenv('DATASETS_FOLDER') + 'pairing.csv')
        knowledge_df=pd.read_csv(os.getenv('DATASETS_FOLDER')+'train_data_prep.csv')
        df_temp = BagPacker.search_knowledge(knowledge_df=knowledge_df,address=df)
        for i in df_temp['breaked_addr']:
            df = df[df['breaked_addr'] != i]
        df_temp = df_temp.append(df)
        df = df_temp
        df1 = df[df['located'] == 0]
        df1, geocode_not_found = BagPacker.get_geocode(df1)
        df = df.append(df1)
        if len(df1)>0:
            BagPacker.update_knowledge_base(knowledge_df=knowledge_df,df=df1)
        df = df[df['located'] == 1]
        df['angles'] = BagPacker.find_angle_on_map(df, BagPacker.find_center(df))
        quad, unmapped, remaining = BagPacker.divide_load(df, 1, len(pkgs))
        # quad, unmapped = BagPacker.divide_load(df, 1)
        quad = BagPacker.distribute_load(quad, 1, 0, len(pkgs))
        for i in range(1):
            if len(quad['quad_%s' % (i + 1)]) > 0:
                data = BagPacker.load_vehicle(quad['quad_%s' % (i + 1)], pair_n=params['pkgs_per_bag'])
                return data['work_pair'], unmapped

    def pack_bags(self, params):
        try:
            #df = pd.read_csv(os.getenv('DATASETS_FOLDER')+'pairing.csv', usecols=['addr', 'pkg_ID',
             #                                                                     'shipment_ID','tag_ID']).dropna()
            df=self.df
            df['shipment_ID']=df['shipment_ID'].astype(str)
            df['located'] = [0 for i in range(len(df))]
            df['breaked_addr'] = BagPacker.clean_address(df['addr'])
            knowledge_df=pd.read_csv(os.getenv('DATASETS_FOLDER')+'train_data_prep.csv')

            #df.to_csv(os.getenv('DATASETS_FOLDER')+'pairing.csv')
            s=timeit.default_timer()
            df_temp = BagPacker.search_knowledge(knowledge_df=knowledge_df,address=df)
            print('search fun:'+str(timeit.default_timer()-s))

            for i in df_temp['breaked_addr']:
                df = df[df['breaked_addr'] != i]
            df = df_temp.append(df,sort=False)

            df1 = df[df['located'] == 0]
            if len(df1) > 0:
                s = timeit.default_timer()
                df1, geocode_not_found = BagPacker.get_geocode(df1)
                print('geocode:'+str(timeit.default_timer()-s))
                df = df.append(df1)
                s = timeit.default_timer()
                BagPacker.update_knowledge_base(knowledge_df=knowledge_df,df=df1)
                print('update'+str(timeit.default_timer()-s))
            else:
                geocode_not_found=[]
            df = df[df['located'] == 1]
            s = timeit.default_timer()
            df['angles'] = BagPacker.find_angle_on_map(df,BagPacker.find_center(df))
            print('angle'+str(timeit.default_timer()-s))
            s = timeit.default_timer()
            quad, unmapped,remaining = BagPacker.divide_load(df, len(params['mobile_hubs']), params['max_cap'])
            print('divide_load:'+str(timeit.default_timer()-s))
            s = timeit.default_timer()
            quad = BagPacker.distribute_load(quad, len(params['mobile_hubs']), params['max_cap']/2, params['max_cap'])
            print('distri:'+str(timeit.default_timer()-s))
            final_result = {}
            if len(remaining) > 0:
                # handling remaining address which are more than eloaders max_cap
                #print(remaining)
                #final_result['remaining'] = dict(mean_pos=dict(loc=[], address=''))
                s = timeit.default_timer()
                final_result['remaining'] = BagPacker.load_vehicle(remaining, pair_n=params['pkgs_per_bag'])
                print('load:'+str(timeit.default_timer()-s))
                #final_result['remaining']['mean_pos']['address'] = BagPacker.get_geocode(pd.DataFrame([
                    #[final_result['remaining']['mean_pos']['loc']]], columns=['loc']), get_address=True)

                #final_result['remaining']['work_pair'] = remaining.values.tolist()

            for i in range(len(params['mobile_hubs'])):
                if len(quad['quad_%s' % (i + 1)]) > 0:
                    s = timeit.default_timer()
                    result = BagPacker.load_vehicle(quad['quad_%s' % (i + 1)], pair_n=params['pkgs_per_bag'])
                    print('load'+str(timeit.default_timer()-s))
                    parking_spot = result['mean_pos']['loc']

                    #result['mean_pos']['address']=BagPacker.get_geocode(pd.DataFrame([[result['mean_pos']['loc']]],columns=['loc']),get_address=True)

                    if params['start_point']:
                        if params['parking_list']:
                            dist = []
                            for j in params['parking_list']:
                                dist.append([math.sqrt((j[0][0] - result['mean_pos']['loc'][0]) ** 2 +
                                                       (j[0][1] - result['mean_pos']['loc'][1])),j[0][0],j[0][1],j[1]])
                                # adding [distance between parking spots and position ideal for eloader,
                                # x coordinate of parking spot,y coordinate of parking spot]
                            parking_spot = []
                            parking_spot.append(sorted(dist)[0][1])
                            parking_spot.append(sorted(dist)[0][2])
                            parking_spot.append(sorted(dist)[0][3])

                            result['parking_spot'] = {'loc':str([parking_spot[0],parking_spot[1]]),'address':parking_spot[2]}
                            dist_time_btw_start_parking = BagPacker.get_short_path([params['start_point'], parking_spot])
                            result['dist_btw_start_parking'] = dist_time_btw_start_parking[0]
                            result['time_btw_start_parking'] = dist_time_btw_start_parking[1]
                            result['max_wait_time'] = (2 * dist_time_btw_start_parking[1]) + (result['max_wait_time'])
                        else:
                            result['max_wait_time'] = (2 * BagPacker.get_short_path(
                                [params['start_point'], result['mean_pos']['loc']])[1]) + result['max_rider_return_time']

                    if 'load_pattern' in params and params['load_pattern']:
                        load_pattern = []
                        for k in result['work_pair']:
                            for j in range(len(k)):
                                load_pattern.append(k[j])

                        result["ordering_pattern"] = BagPacker.determine_loading_pattern(load_pattern,
                                                                                         params['load_pattern'])
                    final_result['%s'%params['mobile_hubs'][i]['name']] = result

                    '''for item in final_result:
                        print (item + ':' + str(len(final_result[item]['work_pair'])) + ' riders with '
                               + str(len(final_result[item]['work_pair']) * params['pkgs_per_bag']) + ' packages')'''

            final_result['unmapped'] = dict(work_pair=unmapped, not_located=geocode_not_found)
            return final_result
        except Exception as e:
            logger.error("Issue with " + str(inspect.stack()[0][3]), exc_info=True)

    @staticmethod
    def bag_of_word_matcher(src_str, dest_str):
        src_str = src_str.split(' ')
        dest_str = dest_str.split(' ')
        count = 0
        for element in src_str:
            if element in dest_str:
                count += 1
        return count / len(src_str)

    @staticmethod
    def fetch_using_ID(use_ID,ID_value):  #USE ID is String type and contains the which ID you want to use i.e pkg_ID,shipment_ID,etc. and ID_value is the 32 character long value of stated ID
        df=pd.read_csv(os.getenv('DATASETS_FOLDER')+'train_data_prep.csv')
        return df[df['%s'%use_ID]==ID_value].to_dict()

    @staticmethod
    def clean_address(addr_list):
        new_breaked_addr = []
        for i in addr_list:
            i = i.lower()
            i = i.replace('(', ' ')
            i = i.replace(')', ' ')
            i = i.replace('-', ' ')
            i = i.replace('[', ' ')
            i = i.replace(']', ' ')
            i = i.replace("'", " ")
            i = i.replace('gurgaon', '')
            i = i.replace('gurugram', '')
            i = i.replace(',', ' ')
            i = ' '.join(filter(None, i.split(' ')))
            i = re.findall(r"[^\W\d_]+|\d+", i)
            temp_addr = ""
            for j in range(len(i)):
                temp_addr += i[j] + " "
            i = temp_addr.strip()
            del temp_addr
            new_breaked_addr.append(i)
        return new_breaked_addr

    @staticmethod
    def get_geocode(df, save_data=False,get_address=False):
        try:
            gmaps = googlemaps.Client(key="AIzaSyCWLiODnzuTOxu9-75Kv7OAaXIKNMJR2rQ")

            if get_address is True:
                address=''
                for i in df['loc']:

                    result=gmaps.reverse_geocode((i[0],i[1]))

                    for addr in result[0]['address_components']:
                        address+=addr['long_name']+' '
                return address


            list_of_address = []
            list_of_lat = []
            list_of_lng = []
            not_located = []
            list_of_located = []
            list_of_pkg_ID = []
            list_of_shipment_ID = []
            list_of_tag_ID = []

            #for i, j, k, l in zip(df['breaked_addr'], df['pkg_ID'],df['shipment_ID'],df['tag_ID']):
            def geocode_multithread(i,j,k,l):
                result = gmaps.geocode('%s,gurugram' % i)
                if not result:
                    not_located.append(i)
                else:
                    list_of_address.append(i)
                    list_of_lat.append(result[0]['geometry']['location']['lat'])
                    list_of_lng.append(result[0]['geometry']['location']['lng'])
                    list_of_pkg_ID.append(j)
                    list_of_shipment_ID.append(k)
                    list_of_tag_ID.append(l)
                    list_of_located.append(1)

            threads=[threading.Thread(target=geocode_multithread, args=(i,j,k,l,)) for i,j,k,l in zip(df['breaked_addr'], df['pkg_ID'],df['shipment_ID'],df['tag_ID'])]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            df = pd.DataFrame(list(zip(list_of_address, list_of_lat, list_of_lng, list_of_located, list_of_pkg_ID,list_of_shipment_ID,list_of_tag_ID)),
                              columns=['breaked_addr', 'x', 'y', 'located', 'pkg_ID','shipment_ID','tag_ID'])
            if save_data:
                df.to_csv("data_from_reverse_geocode.csv")
            return df, not_located
        except Exception as e:
            logger.error("Issue with " + str(inspect.stack()[0][3]), exc_info=True)

    @staticmethod
    def search_knowledge(knowledge_df,address=None,is_block=True):
        try:
            #df = pd.read_csv(os.getenv('DATASETS_FOLDER')+'train_data_prep.csv')
            if address is not None and is_block is False:
                addr = set(address.split(" "))
                for i, j, k, l,m,n in zip(knowledge_df['breaked_addr'], knowledge_df['x_coor'], knowledge_df['y_coor'], knowledge_df['pkg_ID'], knowledge_df['shipment_ID'], knowledge_df['tag_ID']):
                    if set(str(i).split(" ")) == addr:
                        return pd.DataFrame({'breaked_addr': str(i), 'x': [j], 'y': [k], 'pkg_ID': str(l), 'shipment_ID': str(m), 'tag_ID': str(n)})
            if is_block is True:
                list_of_breaked_addr = []
                list_of_x_coor = []
                list_of_y_coor = []
                list_of_located = []
                list_of_pkg_ID = []
                list_of_tag_ID = []
                list_of_shipment_ID = []
                list_of_breaked_addr_set=[]
                for breaked_addr in knowledge_df['breaked_addr']:
                    list_of_breaked_addr_set.append(set(str(breaked_addr).split(' ')))
                #print(list_of_breaked_addr_set)
                knowledge_df['breaked_addr_set']=list_of_breaked_addr_set
                #address = pd.read_csv(os.getenv('DATASETS_FOLDER')+'pairing.csv')

                #for l, m, n, o in zip(address['breaked_addr'], address['pkg_ID'], address['shipment_ID'], address['tag_ID']):
                def search_brain(l,m,n,o):
                    flag = False
                    set_of_input = set(str(l).split(" "))
                    length = len(set(str(l).split(" ")))
                    for i, j, k in zip(knowledge_df['breaked_addr_set'], knowledge_df['x_coor'], knowledge_df['y_coor']):

                        if flag:
                            break
                        else:
                            if len(list(i.intersection(set_of_input))) / length >= 0.85:
                                flag = True
                                list_of_breaked_addr.append(l)
                                list_of_x_coor.append(j)
                                list_of_y_coor.append(k)
                                list_of_located.append(1)
                                list_of_pkg_ID.append(m)
                                list_of_tag_ID.append(o)
                                list_of_shipment_ID.append(n)

                threads=[threading.Thread(target=search_brain,args=(l,m,n,o,)) for l,m,n,o in zip(address['breaked_addr'],address['pkg_ID'],address['shipment_ID'],address['tag_ID'])]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                return pd.DataFrame(list(zip(list_of_breaked_addr, list_of_x_coor, list_of_y_coor, list_of_pkg_ID, list_of_shipment_ID, list_of_tag_ID ,list_of_located)),
                                    columns=['breaked_addr', 'x', 'y','pkg_ID', 'shipment_ID', 'tag_ID' ,'located'])
        except Exception as e:
            logger.error("Issue with " + str(inspect.stack()[0][3]), exc_info=True)

    @staticmethod
    def update_knowledge_base(knowledge_df,df):

        list_of_joint_coordinates = []
        for i, j in zip(df['x'], df['y']):
            list_of_joint_coordinates.append([i, j])

        knowledge_df_new = pd.DataFrame(list(zip(df['breaked_addr'], list_of_joint_coordinates, df['x'], df['y'],
                                                 df['breaked_addr'], ["N/A" for i in range(len(df))], df['pkg_ID'], df['shipment_ID'], df['tag_ID'])),
                                        columns=["addr", "loc", "x_coor", "y_coor", "breaked_addr", "d_hub", "pkg_ID", "shipment_ID", "tag_ID"])
        knowledge_df = knowledge_df.append(knowledge_df_new)
        knowledge_df.to_csv(os.getenv('DATASETS_FOLDER')+'train_data_prep.csv')

    @staticmethod
    def find_center(df):
        city_centre = []  # CITY CENTER OF GURGAON, can also be set to mean point of all the packages [28.457523, 77.026344]
        sum_x = []
        sum_y = []
        for i, j in zip(df['x'], df['y']):
            if int(i) == 28 and (int(j) == 77 or int(j) == 76):
                sum_x.append(float(i))
                sum_y.append(float(j))

        city_centre.append(sum(sum_x) / len(sum_x))
        city_centre.append(sum(sum_y) / len(sum_y))
        return city_centre

    @staticmethod
    def find_angle_on_map(df, center_pt):
        list_of_angles = []  # LIST FOR STORING THE ANGLE FORMATION
        count = 0
        for x, y in zip(df['x'], df['y']):
            if int(x) == 28 and (int(y) == 77 or int(y) == 76):
                dLong = y - center_pt[1]
                count += 1
                dPhi = math.log(math.tan(x / 2.0 + math.pi / 4.0) / math.tan(center_pt[0] / 2.0 + math.pi / 4.0))
                if abs(dLong) > math.pi:
                    if dLong > 0.0:
                        dLong = -(2.0 * math.pi - dLong)
                    else:
                        dLong = (2.0 * math.pi + dLong)

                list_of_angles.append((math.degrees(math.atan2(dLong, dPhi)) + 360.0) % 360.0)
            else:
                list_of_angles.append(0)
        return list_of_angles

    @staticmethod
    def divide_load(df, num_mobile_hubs, max_cap):
        try:
            quad = {}
            for i in range(num_mobile_hubs):
                quad["quad_%s" % (i + 1)] = []
            unmapped = []

            remaining=pd.DataFrame()
            diff = 360 / num_mobile_hubs
            i = 0
            count = 1

            if len(df)>(num_mobile_hubs*max_cap):
                df_temp=df.sort_values(by=['angles'])[max_cap*num_mobile_hubs:]
                df = df.sort_values(by=['angles'])[:max_cap*num_mobile_hubs]
                remaining=df_temp

            while i <= 360:
                for a, b, c, d, e, f, g in zip(df['breaked_addr'], df['x'], df['y'], df['angles'], df['pkg_ID'],df['shipment_ID'],df['tag_ID']):
                    if i <= d < (i + diff):
                        if (int(b) == 28) and (int(c) == 77 or int(c) == 76):
                            quad["quad_%s" % count].append([a, b, c, d, e, f, g])
                        else:
                            unmapped.append({'address':a,'loc':'['+str(b)+','+str(c)+']','pkg_ID': e,'shipment_ID': f,'tag_ID': g})
                count += 1
                i += diff
            return quad, unmapped, remaining
        except Exception as e:
            logger.error("Issue with " + str(inspect.stack()[0][3]), exc_info=True)

    @staticmethod
    def distribute_load(quad, eloader_n, min_cap, max_cap):
        try:
            if eloader_n == 1:
                return quad

            for i in range(eloader_n):
                if len(quad['quad_%s' % (i + 1)]) > max_cap and i is not eloader_n - 1 and len(
                        quad['quad_%s' % (i + 2)]) < max_cap:
                    permit_len = max_cap - len(quad['quad_%s' % (i + 2)])
                    quad['quad_%s' % (i + 2)] += quad['quad_%s' % (i + 1)][:permit_len]
                    quad['quad_%s' % (i + 1)] = quad['quad_%s' % (i + 1)][permit_len:]

                elif len(quad['quad_%s' % (i + 1)]) > max_cap and i is eloader_n - 1 and len(quad['quad_1']) < max_cap:
                    permit_len = max_cap - len(quad['quad_1'])
                    quad['quad_1'] += quad['quad_%s' % (i + 1)][:permit_len]
                    quad['quad_%s' % (i + 1)] = quad['quad_%s' % (i + 1)][permit_len:]

            for i in range(eloader_n):
                if len(quad['quad_%s' % (i + 1)]) > max_cap and i is not 0 and len(quad['quad_%s' % i]) < max_cap:
                    permit_len = max_cap - len(quad['quad_%s' % i])
                    quad['quad_%s' % i] += quad['quad_%s' % (i + 1)][:permit_len]
                    quad['quad_%s' % (i + 1)] = quad['quad_%s' % (i + 1)][permit_len:]

                elif len(quad['quad_1']) > max_cap and i is 0 and len(quad['quad_%s' % (eloader_n - 1)]) < max_cap:
                    permit_len = max_cap - len(quad['quad_%s' % (eloader_n - 1)])
                    quad['quad_%s' % (eloader_n - 1)] += quad['quad_1'][:permit_len]
                    quad['quad_1'] = quad['quad_1'][permit_len:]

            for i in range(eloader_n):
                if len(quad["quad_%s" % (i + 1)]) < min_cap and i is not eloader_n - 1 and len(
                        quad['quad_%s' % (i + 2)]) > min_cap:
                    permit_len = len(quad['quad_%s' % (i + 2)]) - min_cap
                    quad['quad_%s' % (i + 1)] += quad['quad_%s' % (i + 2)][:permit_len]

                    quad['quad_%s' % (i + 2)] = quad['quad_%s' % (i + 2)][permit_len:]

                if len(quad['quad_%s' % (i + 1)]) < min_cap and i is eloader_n - 1 and len(quad['quad_1']) > min_cap:
                    permit_len = len(quad['quad_1']) - min_cap
                    quad['quad_%s' % (eloader_n - 1)] += quad['quad_1'][:permit_len]
                    quad['quad_1'] = quad['quad_1'][permit_len:]

            for i in range(eloader_n):
                if len(quad['quad_%s' % (i + 1)]) < min_cap and i is not 0 and len(quad['quad_%s' % (i)]) > min_cap:
                    permit_len = len(quad['quad_%s' % i]) - min_cap
                    quad['quad_%s' % (i + 1)] += quad['quad_%s' % i][:permit_len:-1]
                    quad['quad_%s' % i] = quad['quad_%s' % i][:permit_len + 1]
                if len(quad['quad_%s' % (i + 1)]) < min_cap and i is 0 and len(
                        quad['quad_%s' % (eloader_n - 1)]) > min_cap:
                    permit_len = len(quad['quad_%s' % (eloader_n - 1)]) - min_cap
                    quad['quad_1'] += quad['quad_%s' % (eloader_n - 1)][:permit_len:-1]
                    quad['quad_%s' % (eloader_n - 1)] = quad['quad_%s' % (eloader_n - 1)][:permit_len + 1]

            for i in range(eloader_n):
                if len(quad['quad_%s' % (i + 1)]) < min_cap and i is not eloader_n - 1:
                    if len(quad['quad_%s' % (i + 2)]) < min_cap:
                        quad['quad_%s' % (i + 2)] += quad['quad_%s' % (i + 1)]
                        quad['quad_%s' % (i + 1)] = []
                if len(quad['quad_%s' % (i + 1)]) < min_cap and i is eloader_n - 1:
                    if (len(quad['quad_1'])) < min_cap:
                        quad['quad_1'] += quad['quad_%s' % (eloader_n - 1)]
                        quad['quad_%s' % (eloader_n - 1)] = []

            for i in range(eloader_n):
                if len(quad['quad_%s' % (eloader_n - i)]) < min_cap and eloader_n - i is not 1:
                    if len(quad['quad_%s' % (eloader_n - i - 1)]) < min_cap:
                        quad['quad_%s' % (eloader_n - i - 1)] += quad['quad_%s' % (eloader_n - i)]
                        quad['quad_%s' % (eloader_n - i)] = []
                if len(quad['quad_%s' % (eloader_n - i)]) < min_cap and eloader_n - i is 1:
                    if (len(quad['quad_%s' % (eloader_n)])) < min_cap:
                        quad['quad_%s' % (eloader_n)] += quad['quad_%s' % (eloader_n - i)]
                        quad['quad_%s' % (eloader_n - i)] = []
            return quad
        except Exception as e:
            logger.error("Issue with " + str(inspect.stack()[0][3]), exc_info=True)

    @staticmethod
    def load_vehicle(df, pair_n, calc_park_spot=False):
        try:
            df1 = pd.DataFrame(df, columns=['breaked_addr', 'x', 'y', 'angles', 'pkg_ID','shipment_ID','tag_ID'])
            work_pair = []


            # STRING MATCHING

            sr_done = []
            list_of_negation = []
            df1['sr'] = [i for i in range(len(df))]

            sum_x = []
            sum_y = []
            for i,j in zip(df1['x'],df1['y']):
                if int(i) == 28 and (int(j) == 77 or int(j) == 76):
                    sum_x.append(i)
                    sum_y.append(j)
            mean_pos = [sum(sum_x) / len(sum_x), sum(sum_y) / len(sum_y)]


            for i, j, k, l, m, n, o in zip(df1['breaked_addr'], df1['sr'], df1['x'], df1['y'], df1['pkg_ID'], df1['shipment_ID'], df1['tag_ID']):
                score = 0
                match_n1 = []
                for x, y, z, a, b, c, d in zip(df1['breaked_addr'], df1['sr'], df1['x'], df1['y'], df1['pkg_ID'], df1['shipment_ID'], df1['tag_ID']):
                    if j != y and (len(list(set(str(x).split(" ")).intersection(str(i).split(" ")))) /
                                   len(list(set(str(x).split(" ")))) >= 0.72) and j not in sr_done and y not in sr_done:
                        if score < (len(list(set(str(x).split(" ")).intersection(str(i).split(" ")))) / len(
                                list(set(str(x).split(" "))))):
                            score = len(list(set(str(i).split(" ")).intersection(str(x).split(" ")))) / len(
                                list(set(str(i).split(" "))))
                            match_n1.append([score, x, y, z, a, b, c, d])

                if score and len(match_n1) >= pair_n - 1:
                    match_n = sorted(match_n1, reverse=True)

                    spare_list = []
                    for index in range(pair_n - 1):
                        spare_list.append(
                            {"pkg_ID":match_n[index][5],"shipment_ID":match_n[index][6],"tag_ID":match_n[index][7],"address": match_n[index][1], "loc": [match_n[index][3], match_n[index][4]]})
                    spare_list.append({"pkg_ID": m,"shipment_ID":n,"tag_ID":o, "address": i, "loc": [k, l]})
                    work_pair.append(spare_list)
                    sr_done.append(j)
                    list_of_negation.append(i)
                    for index in range(len(spare_list) - 1):
                        sr_done.append(match_n[index][2])
                        list_of_negation.append(match_n[index][1])

            for i, j in zip(list_of_negation, sr_done):
                df1 = df1[df1['sr'] != j]

            df = sorted(df1.values.tolist(), key=lambda x: x[3])

            max_wait_time = []
            i = 0
            while i < len(df):
                time_list = []
                if i + pair_n <= len(df) - 1:
                    spare_list = []

                    for j in df[i:i + pair_n]:
                        spare_list.append({"pkg_ID": j[4],"shipment_ID":j[5],"tag_ID":j[6], "address": j[0], "loc": [j[1], j[2]]})
                        time_list.append([j[1], j[2]])

                    work_pair.append(spare_list)
                elif i + pair_n > len(df) - 1:
                    spare_list = []
                    for j in df[i:]:
                        spare_list.append({"pkg_ID": j[4],"shipment_ID":j[5],"tag_ID":j[6], "address": j[0], "loc": [j[1], j[2]]})
                        time_list.append([j[1], j[2]])

                    work_pair.append(spare_list)
                if calc_park_spot:
                    max_wait_time.append(BagPacker.get_short_path(time_list)[1])
                i += pair_n

            return dict(mean_pos={'loc':mean_pos,'address':''}, work_pair=work_pair,
                        max_rider_return_time=(max(max_wait_time) * 2 if max_wait_time else 0))
        except Exception as e:
            logger.error("Issue with " + str(inspect.stack()[0][3]), exc_info=True)

    @staticmethod
    def get_short_path(coords_list=None):
        if len(coords_list) <= 1:
            return [0, 0]

        gmaps = googlemaps.Client(key="AIzaSyBQcyXSSf2I4yGIke4b3ftgPyvcldY-6S8")
        i = 0
        total_m = 0
        total_s = 0
        list_of_travel = []
        while i >= 0 and (i + 1) < len(coords_list):
            list_of_travel.append([gmaps.distance_matrix((coords_list[i][0], coords_list[i][1]),
                                                         (coords_list[i + 1][0], coords_list[i + 1][1]),
                                                         mode='driving')["rows"][0]["elements"][0]["distance"]["value"],
                                   gmaps.distance_matrix((coords_list[i][0], coords_list[i][1]),
                                                         (coords_list[i + 1][0], coords_list[i + 1][1]),
                                                         mode='driving')["rows"][0]["elements"][0]["duration"]["value"]])
            i += 1

        for i in list_of_travel:
            total_m += i[0]
            total_s += i[1]
        return [total_m / 1000, total_s / 60]

    @staticmethod
    def determine_loading_pattern(initial_load, dimensions):
        try:
            input_list = initial_load
            l = dimensions['length']
            w = dimensions['width']
            h = dimensions['height']
            for i in range((w * h * l) - len(input_list)):
                input_list.append(0)
            list_of_packages = np.asarray(input_list, dtype='a20')
            list_of_ordered_packages = np.empty((h, l, b), dtype='a20')
            cur_pkg = -1
            for k in range(h):
                for i in range(l):
                    for j in range(w):
                        cur_pkg += 1
                        if k % 2 == 0:
                            if i % 2 == 0:
                                list_of_ordered_packages[k][i][j] = list_of_packages[cur_pkg]
                            else:
                                list_of_ordered_packages[k][i][w - j - 1] = list_of_packages[cur_pkg]
                        else:
                            if w % 2 != 0:
                                if i % 2 == 0:
                                    list_of_ordered_packages[k][l - i - 1][w - j - 1] = list_of_packages[cur_pkg]
                                else:
                                    list_of_ordered_packages[k][l - i - 1][j] = list_of_packages[cur_pkg]
                            else:
                                if i % 2 == 0:
                                    list_of_ordered_packages[k][l - i - 1][j] = list_of_packages[cur_pkg]
                                else:
                                    list_of_ordered_packages[k][l - i - 1][w - j - 1] = list_of_packages[cur_pkg]
            if (w % 2 != 0 and l % 2 == 0) or (l % 2 != 0 and w % 2 == 0):
                for k in range(h):
                    if k % 2 != 0:
                        for i in range(l):
                            for j in range(int(w / 2)):
                                temp = list_of_ordered_packages[k][i][j]
                                list_of_ordered_packages[k][i][j] = list_of_ordered_packages[k][i][w - j - 1]
                                list_of_ordered_packages[k][i][w - j - 1] = temp

            return list_of_ordered_packages
        except Exception as e:
            logger.error("Issue with " + str(inspect.stack()[0][3]), exc_info=True)


if __name__ == '__main__':
    #pkgs=[{u'address': u'SAURABH CHANDNA H. NO. 862 SECTOR 7 EXTENTION - - GURGAON', u'shipment_ID': u'686517606', u'tag_ID': u'NA', u'ID': u'29b27ff0282a476d97cd2619a9030189'}, {u'address': u'Hno -60sector-14gurgaonharyana, Gurgaon - - - -', u'shipment_ID': u'503984205', u'tag_ID': u'NA', u'ID': u'02cc3aae139d4363bdd0dc4068e31c0c'}, {u'address': u'House nos 958 ,sector 4 , Urba n Estate  (Gurgaon) n Estate  (Gurgaon) - Gurgaon', u'shipment_ID': u'507008693', u'tag_ID': u'NA', u'ID': u'c8440f87fe144c408d1b6fd8d6cd3cb7'}, {u'address': u'1158 sector 4 , urban estate , gurgaon, 1158 secto r 4, gurgaon, - - Gurgaon,', u'shipment_ID': u'507031578', u'tag_ID': u'NA', u'ID': u'1c779f1bb37b4a27a8d3cf573facfc43'}, {u'address': u'434#x2F;6 Near Lal School Jacubpura ,Gurugr Gurgaon Haryana - Gurgaon', u'shipment_ID': u'512612830', u'tag_ID': u'NA', u'ID': u'4fe8338f22f84046a1510a2534a12f40'}, {u'address': u'H.no.957,main shani mandir street , laxman vihar , ph 2 , ggn ,H.no.957, main shani mandir street , laxman vihar ,ph 2 , ggn - Gurgaon', u'shipment_ID': u'512664211', u'tag_ID': u'NA', u'ID': u'c620d8d276234a60910b6bdd3c6f4315'}, {u'address': u'house no 467 ,sector 17 A Gurgaon Haryana - Gurgaon', u'shipment_ID': u'512677945', u'tag_ID': u'NA', u'ID': u'6b31886a845c4de2983b07513a507af0'}, {u'address': u'House no. 636 G, Sector - 15 Part 1 - - - Gurgaon, Haryana', u'shipment_ID': u'526429745', u'tag_ID': u'NA', u'ID': u'5e4c16a1fdf048bc88d7bb5677b2bc4b'}, {u'address': u'house no 77 first floor,sector 17A GURGAON - - - Delhi NCR', u'shipment_ID': u'526430741', u'tag_ID': u'NA', u'ID': u'7fd967d9246d4aa0ab77537d22ad55d9'}, {u'address': u'OASIS GURGAON, B-11 SECOND FLOOR, OLD DLF COLONY, SEC-14, GURGAON - - GURGAON', u'shipment_ID': u'526442553', u'tag_ID': u'NA', u'ID': u'ebfe4bef2c9846488eb428fbb5040a9e'}, {u'address': u'Shop no-37 Behind bharat petrol pump,12 biswa road 12 biswa road, - - Gurgaon', u'shipment_ID': u'528260544', u'tag_ID': u'NA', u'ID': u'fdcf75d8ad664fdd93a788a5372afda7'}, {u'address': u'Shop no-37 Behind bharat petrol pump,12 biswa road 12 biswa road, - - Gurgaon', u'shipment_ID': u'528299438', u'tag_ID': u'NA', u'ID': u'417b0928786f45e08d7f9edd11809175'}, {u'address': u'HOUSE NO-48,12 BISWA ROAD,SARAI WALA RASTA,NEAR RA Gurgaon - - Gurgaon', u'shipment_ID': u'528317859', u'tag_ID': u'NA', u'ID': u'dd9e93e81a3645fabbda974633ab3fc7'}, {u'address': u'1534 first floor back side sector 17c 1534 first floor back side sector 17c - - Gurgaon', u'shipment_ID': u'529781901', u'tag_ID': u'NA', u'ID': u'858ce5c76f5c44df9d622dd7525a5d27'}, {u'address': u'Shop no-37 Behind bharat petrol pump,12 biswa road 12 biswa road, - - Gurgaon', u'shipment_ID': u'528435966', u'tag_ID': u'NA', u'ID': u'65e56f75789f450b9afc6d50bd1f8e58'}, {u'address': u'Shop no-37 Behind bharat petrol pump,12 biswa road 12 biswa road, - - Gurgaon', u'shipment_ID': u'528552960', u'tag_ID': u'NA', u'ID': u'd13bed7b451947549d75c10f8a7bfc59'}, {u'address': u'Shop no-37 Behind bharat petrol pump,12 biswa road 12 biswa road, - - Gurgaon', u'shipment_ID': u'528580253', u'tag_ID': u'NA', u'ID': u'281b0f0c00804da0b7f5b3fe15cdb042'}, {u'address': u'H.No.899, Basement, Sector 4, Behind Allahabad Ba Behind Allahabad Bank - - Gurgaon', u'shipment_ID': u'528684197', u'tag_ID': u'NA', u'ID': u'33ca9e58bbc5494e8e0759b2fd2820d6'}, {u'address': u'ANUJ 437,SECTOR 17 A - - GURUGRAM', u'shipment_ID': u'686715187', u'tag_ID': u'NA', u'ID': u'68fae36c77ac4088b6632ddf519c3b91'}, {u'address': u'DHANANJAY SHARMA 148, SECOND FLOOR, WEWORK,,HOT EL GAL AXY, SECTOR 15, PART-1, - GURUGRAM', u'shipment_ID': u'686758818', u'tag_ID': u'NA', u'ID': u'32c337e3c3e445f5a69653175ab4315a'}, {u'address': u'SURUCHI JINDAL 618 , SECTOR 15 PART 1 - - GURGAON', u'shipment_ID': u'686800302', u'tag_ID': u'NA', u'ID': u'32cbc276fc07462a85a3a8b84b034afe'}, {u'address': u'AMIT GARG NEAR ROHILLA DHARMSALA, H.NO 1918 GALI NO 3 RAJIV NAGER - GURGAON', u'shipment_ID': u'686835533', u'tag_ID': u'NA', u'ID': u'32421347d3fa4d39afe73665c4581cb6'}, {u'address': u'SANJEEV MISHRA 33512, SECOND FLOOR, CHIRAG HOUSE,SUBHAS NAGAR, NEAR - GURUGRAM', u'shipment_ID': u'686842555', u'tag_ID': u'NA', u'ID': u'097f65e5a3184f33ab79f6c6cbfca9f7'}, {u'address': u'KAVITA AASHISH YADAV 373B,LANE 5AAMANPURA,SHEETLA MATA MANDIR - GURUGRAM', u'shipment_ID': u'686863840', u'tag_ID': u'NA', u'ID': u'60aa294ade454d6cb3b01472290bec54'}, {u'address': u'DHARMENDER SACHDEVA 1027, SECTOR 15, PART II, , MAIN ROAD - GURGAON', u'shipment_ID': u'686878585', u'tag_ID': u'NA', u'ID': u'417628b0d4bd44d8bdb2432a6513c045'}, {u'address': u'BODHIFAB PVT LTD HOUSE 203, DAHIYA GALI SUKHRAL I, SECTOR 17ASECTOR 17A NEAR - GURUGRAM', u'shipment_ID': u'686917185', u'tag_ID': u'NA', u'ID': u'd653cb238d194c5c8e3e16dc5179a631'}, {u'address': u'SUNNY KUMAR GALI. NO-8, HOUSE. NO. -22, BLOCK-C, SHEETLA COLONYSHEETL - GURUGRAM', u'shipment_ID': u'686931198', u'tag_ID': u'NA', u'ID': u'cb30de9231c94c7e84c5416783c1d273'}, {u'address': u'ANKUR ARORA H.NO 200 FIRST FLOOR SECTOR 14NEAR DAV SCHOOL - GURUGRAM', u'shipment_ID': u'686943726', u'tag_ID': u'NA', u'ID': u'6e6669fec19045e3b6d62423ba752509'}, {u'address': u'NM-13, 2nd Floor, Old Dlf Sector14 Gurgaon - - Gurgaon', u'shipment_ID': u'528253799', u'tag_ID': u'NA', u'ID': u'c95d7f5e0794498d95a794a35d03bd20'}, {u'address': u'NM-13, 2nd Floor, Old Dlf Sector14 Gurgaon - - Gurgaon', u'shipment_ID': u'528257766', u'tag_ID': u'NA', u'ID': u'd29f7408037b48778a3f4018378136c3'}, {u'address': u'NM-13, 2nd Floor, Old Dlf Sector14 Gurgaon - - Gurgaon', u'shipment_ID': u'528319815', u'tag_ID': u'NA', u'ID': u'5e432f6efc7c4f56a2cfb735a263befa'}, {u'address': u'NM-13, 2nd Floor, Old Dlf Sector14 Gurgaon - - Gurgaon', u'shipment_ID': u'528510373', u'tag_ID': u'NA', u'ID': u'f4127c94e6c64ae881de648f3047f79a'}, {u'address': u'NM-13, 2nd Floor, Old Dlf Sector14 Gurgaon - - Gurgaon', u'shipment_ID': u'528567988', u'tag_ID': u'NA', u'ID': u'c3f570cd3fbd498fa43cf061b3726ed8'}, {u'address': u'107R Near Dev Samaj School  New Colony Gurgaon Har yana Gurgaon - Gurgaon', u'shipment_ID': u'550435976', u'tag_ID': u'NA', u'ID': u'ccab2577fff5476a90811bfb981f61c4'}, {u'address': u'107R Near Dev Samaj School New Colony Gurgao - - GURGAON', u'shipment_ID': u'550442738', u'tag_ID': u'NA', u'ID': u'd33fab499d574fa78a86aa1dc90b2cad'}]
    pkgs = [{u'address': u'House nos 958 ,sector 4 , Urba n Estate  (Gurgaon) n Estate  (Gurgaon) - Gurgaon',
             u'shipment_ID': u'507008693', u'tag_ID': u'D441C9071603', u'ID': u'c8440f87fe144c408d1b6fd8d6cd3cb7'},
            {u'address': u'H.No. 899, Basement, Sector 4, Behind Allahabad Ba Behind Allahabad Bank - - Gurgaon',
             u'shipment_ID': u'528684197', u'tag_ID': u'BARCODE', u'ID': u'33ca9e58bbc5494e8e0759b2fd2820d6'},
            {u'address': u'AMIT GARG NEAR ROHILLA DHARMSALA, H.NO 1918 GALI NO 3 RAJIV NAGER - GURGAON',
             u'shipment_ID': u'686835533', u'tag_ID': u'BARCODE', u'ID': u'32421347d3fa4d39afe73665c4581cb6'}, {
                u'address': u'500  9, Basement, Shiv Puri, Circular Road Adjoining to BACHPAN Kids Play School - - Gurgaon',
                u'shipment_ID': u'528628757', u'tag_ID': u'0', u'ID': u'afb646ebed4743ff905ab76a5b3c1d52'},
            {u'address': u'57028 jyoti park Gurgaon gali no 10 57028 jyoti park Gurgaon gali no 10 - - Gurgaon',
             u'shipment_ID': u'529826374', u'tag_ID': u'D441C9071603', u'ID': u'9dbf332dd24143f998785e0c82b71549'},
            {u'address': u'15 Choti Mata Mandir gali shukharli 15 Choti Mata Mandir gali shukharli - - Gurgaon',
             u'shipment_ID': u'529840257', u'tag_ID': u'D441C9071603', u'ID': u'b59d100a5a0b45b484f196bafa1f8d2b'},
            {u'address': u'PATRICK P RAY (PPR) HOUSE NO. 46, FLOOR 2ND, ROOM NO#3  CHOPAL WALA GALLI, - GURUGRAM',
             u'shipment_ID': u'687016085', u'tag_ID': u'0', u'ID': u'f6578d996eb343cd93093f02ee1ca08e'},
            {u'address': u'house no 467 ,sector 17 A Gurgaon Haryana - Gurgaon', u'shipment_ID': u'512694867',
             u'tag_ID': u'0', u'ID': u'81a5214f93f24c70a424279c20af1ade'},
            {u'address': u'1389 shivpuri behind ganga motors . - - Gurgaon', u'shipment_ID': u'523451718',
             u'tag_ID': u'0', u'ID': u'36fd5d0208e0466b8e4e914c3d4abab7'},
            {u'address': u'H.No 271,West Rajeev nagar Gurgaon Haryana - Gurgaon', u'shipment_ID': u'526227761',
             u'tag_ID': u'0', u'ID': u'480fb5cab5814df1be9dab0dfd67ef2f'}, {
                u'address': u'Khushal ayurvedicstore,New railway road opposite s tate bank ofpatiyala near auto stand Haryana - Gurgaon',
                u'shipment_ID': u'526228713', u'tag_ID': u'0', u'ID': u'3c9407f28db94380ac2a3b903784d182'},
            {u'address': u'321 sector 7 - - - Gurgaon', u'shipment_ID': u'526435135', u'tag_ID': u'0',
             u'ID': u'7f4957f19edd47b48b1e81afb99aa31e'},
            {u'address': u'House No.139-Y9, Shivpuri Gurgaon-122001 Landmar Gurgaon - - Gurgaon',
             u'shipment_ID': u'528487095', u'tag_ID': u'0', u'ID': u'6bcbec6ebd414f00b7cada7f8d4da93a'},
            {u'address': u'B-204, SURYA VIHAR, SECTOR -4  GURGAON NEAR BY 47 Gurgaon - - Gurgaon',
             u'shipment_ID': u'528577840', u'tag_ID': u'0', u'ID': u'3462876f81cd43048ed7be7c94779a9f'},
            {u'address': u'hn 17 gali no 7 block d sheetla clony gurgaon gur aon - - Gurgaon',
             u'shipment_ID': u'529892300', u'tag_ID': u'0', u'ID': u'6125b5f4b1184bd8b0f8fd432a2a1b3c'},
            {u'address': u'c 120,sector 14 Gurgaon Haryana - Gurgaon', u'shipment_ID': u'532668794', u'tag_ID': u'0',
             u'ID': u'd253e0c1cab64b8eaa3474046c3363f5'},
            {u'address': u'Richa Global export Pvt ltd sec 7 plot no 407,bas khusal Haryana - Gurgaon',
             u'shipment_ID': u'532673460', u'tag_ID': u'0', u'ID': u'c6b02507b673431ab1b7e4065a51bc5d'}, {
                u'address': u'c 80 , old dlf colony , near sector 14,c 80 , old dlf colony , near sector 14 Haryana - Gurgaon',
                u'shipment_ID': u'532713647', u'tag_ID': u'0', u'ID': u'350b57a394ca4325a0976f1270714a01'},
            {u'address': u'12443 back side gali no.5rajiv nagar Tel:98732395 66 - - gurgaon',
             u'shipment_ID': u'900275673', u'tag_ID': u'0', u'ID': u'9a40c5c543c74688ab2219bc7dc72611'},
            {u'address': u'613Sector 14 - - - -', u'shipment_ID': u'513970335', u'tag_ID': u'0',
             u'ID': u'372f7e33d4c94ffaa7e81644e644ed3a'},
            {u'address': u'B - 263 Mianwali Colony - - - Gurgaon', u'shipment_ID': u'526438051', u'tag_ID': u'0',
             u'ID': u'502cc18354fd400cab93065b452c5972'},
            {u'address': u'dilli light shiv Puri dilli light shiv Puri - - Gurgaon', u'shipment_ID': u'529811370',
             u'tag_ID': u'0', u'ID': u'a8ada71bffe44f88bae3d9f8126db77c'},
            {u'address': u'256, Basement, Near Gupta Hospital, Jacobpura Gurgaon - - Gurgaon',
             u'shipment_ID': u'528440038', u'tag_ID': u'0', u'ID': u'da7b845244d74091aa5e307a066a6276'}]
    #os.environ["DATASETS_FOLDER"] = os.environ['PWD'] + "/datasets/"
    os.environ["DATASETS_FOLDER"] = os.getcwd() + "/../datasets/"

    info = dict(mobile_hubs=[{'name':'1'},{'name':'2'}], max_cap=10, pkgs_per_bag=15, start_point=[28.4861289, 77.0620486], parking_list=None)

    start_time = timeit.default_timer()
    print(BagPacker(pkgs).pack_bags(info))
    elapsed = timeit.default_timer() - start_time
    print('ELAPSED TIME FOR ENTIRE CODE:'+str(elapsed)+'s')
