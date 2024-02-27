# color_list = [
#     (0, 0, 0),  # sea
#
#     # HRSC2016
#     # (255, 0, 0),  # land
#     # (0, 255, 0)  # ship
#
#     # Kaggle
#     (128, 0, 0),  # land
#     (0, 128, 0)  # ship
# ]


def select_color_list(dataset_name):
    if dataset_name == "HRSC2016DS":
        return [
            (0, 0, 0),  # sea
            (255, 0, 0),  # land
            (0, 255, 0)  # ship
        ]
    elif dataset_name == "KaggleLandShip":
        return [
            (0, 0, 0),  # sea
            (128, 0, 0),  # land
            (0, 128, 0)  # ship
        ]
