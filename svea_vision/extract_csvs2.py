import rosbag
import pandas as pd
import os
import rospy_message_converter.message_converter as converter

CURRENT_FILE = os.getcwd()
file_paths = ["out_2024-03-04-19-08-30_one_person_moving", "out_2024-03-04-19-09-45_standing", 
              "out_2024-03-04-19-10-28_two_people_moving", "out_2024-03-04-19-11-38_one_person_standing_one_moving", 
              "out_2024-03-04-19-13-21"]


def create_dir(dir):
    """
    Input:
        dir: a directory to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


# Function to flatten the dictionary
def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def save_data(results_path, data_path):

    # The bag file should be in the same directory as your terminal
    bag = rosbag.Bag(data_path + '.bag')
    topics = ['/objectposes', '/person_state_estimation/person_states', '/qualisys/pedestrian/pose', '/qualisys/pedestrian/velocity', '/qualisys/tinman/pose', '/qualisys/tinman/velocity']
    create_dir(results_path)

    for topic, msg, t in bag.read_messages(topics=topics):
        d = converter.convert_ros_message_to_dictionary(msg)
        print(d)
        print(topic)
        print('\n')

        dataset = pd.DataFrame([flatten_dict(d)])    
        data_file_path = results_path + topic.replace('/', '_') + '.csv'    
        dataset.to_csv(data_file_path, columns=dataset.keys())


if __name__ == "__main__":
    for file_name in file_paths:
        results_path = CURRENT_FILE + '/results/' + file_name + '/'
        data_path = CURRENT_FILE + '/' + file_name
        save_data(results_path, data_path)
