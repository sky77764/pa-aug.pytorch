import numpy as np
import box_np_ops

def get_partition_corners(augmented_box_corners, partition_idx, gt_names):
    partition_corners = []
    if gt_names == "Car":
        if partition_idx == 0:
            corner_keys = ['0', '01', '02', '03', '04', '05', '06', '07']
        elif partition_idx == 1:
            corner_keys = ['01', '1', '12', '02', '05', '15', '16', '06']
        elif partition_idx == 2:
            corner_keys = ['02', '12', '2', '23', '06', '16', '26', '27']
        elif partition_idx == 3:
            corner_keys = ['03', '02', '23', '3', '07', '06', '27', '37']
        elif partition_idx == 4:
            corner_keys = ['04', '05', '06', '07', '4', '45', '46', '47']
        elif partition_idx == 5:
            corner_keys = ['05', '15', '16', '06', '45', '5', '56', '46']
        elif partition_idx == 6:
            corner_keys = ['06', '16', '26', '27', '46', '56', '6', '67']
        elif partition_idx == 7:
            corner_keys = ['07', '06', '27', '37', '47', '46', '67', '7']
        else:
            print("ERROR: wrong partition_idx {}".format(partition_idx))
            exit()
    elif gt_names == "Pedestrian":
        if partition_idx == 0:
            corner_keys = ['0', '01', '23', '3', '04', '05', '27', '37']
        elif partition_idx == 1:
            corner_keys = ['01', '1', '2', '23', '05', '15', '26', '27']
        elif partition_idx == 2:
            corner_keys = ['05', '15', '26', '27', '45', '5', '6', '67']
        elif partition_idx == 3:
            corner_keys = ['04', '05', '27', '37', '4', '45', '67', '7']
        else:
            print("ERROR: wrong partition_idx {}".format(partition_idx))
            exit()
    elif gt_names == "Cyclist":
        if partition_idx == 0:
            corner_keys = ['0', '01', '02', '03', '4', '45', '46', '47']
        elif partition_idx == 1:
            corner_keys = ['01', '1', '12', '02', '45', '5', '56', '46']
        elif partition_idx == 2:
            corner_keys = ['02', '12', '2', '23', '46', '56', '6', '67']
        elif partition_idx == 3:
            corner_keys = ['03', '02', '23', '3', '47', '46', '67', '7']
        else:
            print("ERROR: wrong partition_idx {}".format(partition_idx))
            exit()
    else:
        print("ERROR: wrong gt_names {}".format(gt_names))

    for key in corner_keys:
        partition_corners.append(augmented_box_corners[key])
    partition_corners = np.expand_dims(np.array(partition_corners), axis=0)
    return partition_corners


def augment_box_corners(box_corners):
    augmented_box_corners = {}
    for i, corner in enumerate(box_corners):
        augmented_box_corners[str(i)] = corner
    augmented_box_corners["01"] = (box_corners[0] + box_corners[1]) / 2
    augmented_box_corners["02"] = (box_corners[0] + box_corners[2]) / 2
    augmented_box_corners["03"] = (box_corners[0] + box_corners[3]) / 2
    augmented_box_corners["04"] = (box_corners[0] + box_corners[4]) / 2
    augmented_box_corners["05"] = (box_corners[0] + box_corners[5]) / 2
    augmented_box_corners["06"] = (box_corners[0] + box_corners[6]) / 2
    augmented_box_corners["07"] = (box_corners[0] + box_corners[7]) / 2
    augmented_box_corners["12"] = (box_corners[1] + box_corners[2]) / 2
    augmented_box_corners["15"] = (box_corners[1] + box_corners[5]) / 2
    augmented_box_corners["16"] = (box_corners[1] + box_corners[6]) / 2
    augmented_box_corners["23"] = (box_corners[2] + box_corners[3]) / 2
    augmented_box_corners["26"] = (box_corners[2] + box_corners[6]) / 2
    augmented_box_corners["27"] = (box_corners[2] + box_corners[7]) / 2
    augmented_box_corners["37"] = (box_corners[3] + box_corners[7]) / 2
    augmented_box_corners["45"] = (box_corners[4] + box_corners[5]) / 2
    augmented_box_corners["46"] = (box_corners[4] + box_corners[6]) / 2
    augmented_box_corners["47"] = (box_corners[4] + box_corners[7]) / 2
    augmented_box_corners["56"] = (box_corners[5] + box_corners[6]) / 2
    augmented_box_corners["67"] = (box_corners[6] + box_corners[7]) / 2
    return augmented_box_corners


def assign_box_points_partition(points, gt_boxes, gt_names, num_partition):
    assert len(points.shape) == 2, 'Wrong points.shape'
    box_points_mask, box_corners_3d = box_np_ops.points_corners_in_rbbox(points, gt_boxes)
    bg_mask = np.logical_not(np.any(box_points_mask, axis=1))
    assert len(box_points_mask.shape) == 2, 'Wrong box_points_mask.shape'

    fg_seperated_points = []
    partition_corners_list = []
    for i in range(box_points_mask.shape[1]):
        separated_box_points = []
        box_points = points[box_points_mask[:, i]]
        box_corners = box_corners_3d[i]
        augmented_box_corners = augment_box_corners(box_corners)
        partition_corners_np = np.zeros((num_partition[gt_names[i]], 8, 3))
        for j in range(num_partition[gt_names[i]]):
            partition_corners = get_partition_corners(augmented_box_corners, j, gt_names[i])   # [1, 8, 3]
            partition_corners_np[j] = partition_corners
            partition_points_mask = np.squeeze(box_np_ops.points_in_corners(box_points, partition_corners), axis=1)
            assert len(box_points[partition_points_mask].shape) == 2, 'Wrong box_points[partition_points_mask].shape'
            separated_box_points.append(box_points[partition_points_mask])
        partition_corners_list.append(partition_corners_np)
        fg_seperated_points.append(separated_box_points)

    bg_points = points[bg_mask]
    return fg_seperated_points, bg_points, partition_corners_list

def get_random_boxes(gt_boxes, gt_names, num_partition, box_corners_3d):
    dims = gt_boxes[:, 3:6]
    rys = gt_boxes[:, 6]

    corners_pair = [[1, 2], [1, 5], [1, 0]]
    corners_pair_opposite = [[5, 6], [2, 6], [6, 7]]

    random_partitions = []
    for i in range(dims.shape[0]):
        l = np.linalg.norm(box_corners_3d[i, corners_pair[0][0], :] - box_corners_3d[i, corners_pair[0][1], :])
        w = np.linalg.norm(box_corners_3d[i, corners_pair[1][0], :] - box_corners_3d[i, corners_pair[1][1], :])
        h = np.linalg.norm(box_corners_3d[i, corners_pair[2][0], :] - box_corners_3d[i, corners_pair[2][1], :])

        random_partition = []
        for j in range(num_partition[gt_names[i]]):
            t_list = []
            min_diff = 0.4
            for k in range(3):
                t = np.random.uniform(low=0, high=1, size=(1, 2))
                if gt_names[i] == "Car":
                    while True:
                        t = np.random.uniform(low=0, high=1, size=(1, 2))
                        if np.abs(t[0, 0] - t[0, 1]) >= min_diff:
                            break
                t_list.append(t)

            t_list = np.concatenate(t_list, axis=0)
            partition_l = l * np.abs(t_list[0][0] - t_list[0][1])
            partition_w = w * np.abs(t_list[1][0] - t_list[1][1])
            partition_h = h * np.abs(t_list[2][0] - t_list[2][1])

            t_mean = np.mean(t_list, axis=1)
            point_l0 = t_mean[0] * box_corners_3d[i, corners_pair[0][0], :] + (1 - t_mean[0]) * box_corners_3d[i, corners_pair[0][1], :]
            point_l1 = t_mean[0] * box_corners_3d[i, corners_pair_opposite[0][0], :] + (1 - t_mean[0]) * box_corners_3d[i, corners_pair_opposite[0][1], :]

            point_top = t_mean[1] * point_l0 + (1 - t_mean[1]) * point_l1
            point_h = t_mean[2] * box_corners_3d[i, corners_pair[2][0], :] + (1 - t_mean[2]) * box_corners_3d[i, corners_pair[2][1], :]

            random_partition.append([point_top[0], point_top[1], point_h[2], partition_w, partition_l, partition_h, rys[i]])

        random_partitions.append(random_partition)

    return random_partitions

def assign_random_partition(points, gt_boxes, gt_names, num_partition):
    assert len(points.shape) == 2, 'Wrong points.shape'
    box_points_mask, box_corners_3d = box_np_ops.points_corners_in_rbbox(points, gt_boxes)
    bg_mask = np.logical_not(np.any(box_points_mask, axis=1))
    assert len(box_points_mask.shape) == 2, 'Wrong box_points_mask.shape'

    random_partitions = get_random_boxes(gt_boxes, gt_names, num_partition, box_corners_3d)

    fg_seperated_points = []
    fg_not_assigned_points = []
    partition_corners_list = []
    for i in range(box_points_mask.shape[1]):
        separated_box_points = []
        box_points = points[box_points_mask[:, i]]
        partition_points_mask = np.zeros((box_points.shape[0]), dtype=np.bool)
        random_partition = np.array(random_partitions[i])
        partition_corners = box_np_ops.center_to_corner_box3d(random_partition[:, 0:3], random_partition[:, 3:6], random_partition[:, 6], origin=(0.5, 0.5, 0.5), axis=2)
        partition_corners_list.append(partition_corners)
        for j in range(num_partition[gt_names[i]]):
            not_assigned_points_mask = np.logical_not(partition_points_mask)
            cur_box_points = box_points[not_assigned_points_mask]
            cur_partition_mask = np.squeeze(box_np_ops.points_in_corners(cur_box_points, partition_corners[j:j+1]), axis=1)
            assert len(box_points[partition_points_mask].shape) == 2, 'Wrong box_points[partition_points_mask].shape'
            separated_box_points.append(cur_box_points[cur_partition_mask])
            partition_points_mask[not_assigned_points_mask] = np.logical_or(partition_points_mask[not_assigned_points_mask], cur_partition_mask)
        fg_seperated_points.append(separated_box_points)
        fg_not_assigned_points.append(box_points[np.logical_not(partition_points_mask)])


    bg_points = points[bg_mask]
    check_empty = True
    for p in fg_not_assigned_points:
        if p.shape[0] > 0:
            check_empty = False
    if not check_empty:
        fg_not_assigned_points = np.concatenate(fg_not_assigned_points, axis=0)
        bg_points = np.concatenate([bg_points, fg_not_assigned_points], axis=0)
    return fg_seperated_points, bg_points, partition_corners_list, random_partitions

def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)

def farthest_point_sampling(pts, K):
    farthest_pts = np.zeros((K, 3))
    farthest_pts_idx = np.zeros(K, dtype=np.int)
    farthest_pts_idx[0] = np.random.randint(len(pts))
    farthest_pts[0] = pts[farthest_pts_idx[0]]

    distances = calc_distances(farthest_pts[0], pts)

    for i in range(1, K):
        farthest_pts_idx[i] = np.argmax(distances)
        farthest_pts[i] = pts[farthest_pts_idx[i]]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts_idx

class PartAwareAugmentation(object):
    def __init__(self, points, gt_boxes, gt_names, class_names=None, random_partition=False):
        self.points = points
        self.gt_boxes = gt_boxes
        self.gt_names = gt_names
        self.num_gt_boxes = gt_boxes.shape[0]
        self.gt_boxes_mask = [True] * self.num_gt_boxes
        self.num_partition = {}
        self.num_partition['Car'] = 8
        self.num_partition['Pedestrian'] = 4
        self.num_partition['Cyclist'] = 4
        self.num_classes = len(class_names)
        self.random_partition = random_partition
        if random_partition:
            self.separated_box_points, self.bg_points, self.partition_corners, self.random_partition_boxes = \
                assign_random_partition(self.points, gt_boxes, gt_names, self.num_partition)
        else:
            self.separated_box_points, self.bg_points, self.partition_corners = assign_box_points_partition(self.points, gt_boxes, gt_names, self.num_partition)
        self.aug_flag = np.zeros((self.num_gt_boxes, 8, 6), dtype=np.bool)

    def interpret_pa_aug_param(self, pa_aug_param):
        pa_aug_param_dict={}
        method_list = ['dropout', 'sparse', 'noise', 'swap', 'mix', 'jitter', 'random', 'distance']
        for method in method_list:
            if method == 'distance':
                pa_aug_param_dict[method] = 100
                continue
            if method == 'random':
                pa_aug_param_dict[method] = False
                continue
            pa_aug_param_dict[method] = 0
            pa_aug_param_dict[method + '_p'] = 0

        if pa_aug_param is None:
            return pa_aug_param_dict

        param_list = pa_aug_param.split('_')
        for i, param in enumerate(param_list):
            if param.startswith('p'):
                continue
            for method in method_list:
                if param.startswith(method):
                    if method == 'random':
                        pa_aug_param_dict[method] = True
                        method_list.remove(method)
                        break
                    number = param.replace(method, "")
                    if len(number) == 0:
                        if method == "jitter":
                            pa_aug_param_dict[method] = 0.1
                        else:
                            pa_aug_param_dict[method] = 1
                    else:
                        if method == "jitter":
                            pa_aug_param_dict[method] = float(number) / 10 ** (len(number) - 1)
                        else:
                            pa_aug_param_dict[method] = int(number)
                    if method == 'distance':
                        method_list.remove(method)
                        break
                    pa_aug_param_dict[method + '_p'] = 1.0
                    if param_list[i + 1].startswith('p'):
                        number = param_list[i + 1].replace('p', "")
                        if len(number) == 2:
                            pa_aug_param_dict[method + '_p'] = float(number) / 10.0
                        elif len(number) == 3:
                            pa_aug_param_dict[method + '_p'] = float(number) / 100.0
                    method_list.remove(method)
                    break

        return pa_aug_param_dict

    def remove_empty_gt_boxes(self):
        for i in range(self.num_gt_boxes):
            check_empty = True
            if self.random_partition:
                box_points_mask, _ = box_np_ops.points_corners_in_rbbox(self.points, self.gt_boxes[i:i+1])
                if np.any(box_points_mask, axis=0)[0]:
                    check_empty = False
            else:
                for j in range(self.num_partition[self.gt_names[i]]):
                    if self.separated_box_points[i][j].shape[0] > 0:
                        check_empty = False
                        break
            if check_empty:
                self.gt_boxes_mask[i] = False

        self.gt_boxes = self.gt_boxes[self.gt_boxes_mask]
        self.gt_names = self.gt_names[self.gt_boxes_mask]
        self.num_gt_boxes = self.gt_boxes.shape[0]
        self.separated_box_points = [d for d, s in zip(self.separated_box_points, self.gt_boxes_mask) if s]
        self.aug_flag = self.aug_flag[self.gt_boxes_mask]

    def stack_fg_points(self):
        fg_points = np.zeros((0, 4))
        for i in range(self.num_gt_boxes):
            for j in range(self.num_partition[self.gt_names[i]]):
                fg_points = np.vstack((fg_points, self.separated_box_points[i][j]))
        return fg_points

    def stack_fg_points_idx(self, idx=0):
        fg_points = np.zeros((0, 4))
        for i in range(self.num_gt_boxes):
            if i == idx:
                for j in range(self.num_partition[self.gt_names[i]]):
                    fg_points = np.vstack((fg_points, self.separated_box_points[idx][j]))
        return fg_points

    def stack_fg_points_mask(self):
        fg_points = np.zeros((0, 4))
        for i in range(self.num_gt_boxes):
            if self.gt_boxes_mask[i]:
                for j in range(self.num_partition[self.gt_names[i]]):
                    fg_points = np.vstack((fg_points, self.separated_box_points[i][j]))
        return fg_points

    #####################################
    # remove points in random sub-parts #
    #####################################
    def dropout_partitions(self, num_dropout_partition=4, distance_limit=100, p=1.0, gt_box_idx=None, robustness_test=False):
        if num_dropout_partition <= 0:
            return

        for i in range(self.num_gt_boxes):
            if gt_box_idx is not None and i != gt_box_idx:
                continue
            if self.gt_boxes[i][0] > distance_limit:
                continue
            if np.random.rand(1) > p:
                continue
            if robustness_test:
                max_num_points = 0
                for j, separated_box_points in enumerate(self.separated_box_points[i]):
                    if separated_box_points.shape[0] >= max_num_points:
                        max_num_points = separated_box_points.shape[0]
                        dropout_partition_idxes = [j]
                print(max_num_points)

            else:
                dropout_partition_idxes = np.random.choice(range(self.num_partition[self.gt_names[i]]), num_dropout_partition, replace=False)
            for j in range(len(dropout_partition_idxes)):
                self.separated_box_points[i][dropout_partition_idxes[j]] = np.zeros((0, 4))
                self.aug_flag[i, dropout_partition_idxes[j], 0] = True
        self.remove_empty_gt_boxes()


    #######################################
    # swap points in non-empty partitions #
    #######################################
    def swap_partitions(self, num_swap=1, distance_limit=100, p=1.0, gt_box_idx=None):
        if num_swap <= 0:
            return

        for i in range(self.num_gt_boxes):
            if gt_box_idx is not None and i != gt_box_idx:
                continue
            if self.gt_boxes[i][0] > distance_limit:
                continue
            if np.random.rand(1) > p:
                continue
            gt_idxes = list(range(self.num_gt_boxes))

            if self.num_classes > 1:
                # remove idxes of different classes
                same_class_mask = self.gt_names == self.gt_names[i]
                same_class_mask[i] = False
                gt_idxes = [i for (i, m) in zip(gt_idxes, same_class_mask) if m]
            else:
                gt_idxes.remove(i)

            # find non-empty partition idx for both gt
            non_empty_partition_idx = -1
            while len(gt_idxes) > 0:
                target_gt_idx = np.random.choice(gt_idxes, 1, replace=False)[0]
                non_empty_partition_idxes = [idx for idx, points in enumerate(self.separated_box_points[i]) if
                                             points.shape[0] != 0]
                while len(non_empty_partition_idxes) > 0:
                    non_empty_partition_idx_candidate = \
                    np.random.choice(non_empty_partition_idxes, 1, replace=False)[0]
                    if self.separated_box_points[target_gt_idx][non_empty_partition_idx_candidate].shape[0] != 0:
                        non_empty_partition_idx = non_empty_partition_idx_candidate
                        break
                    non_empty_partition_idxes.remove(non_empty_partition_idx_candidate)
                if non_empty_partition_idx != -1:
                    break
                gt_idxes.remove(target_gt_idx)

            if non_empty_partition_idx == -1:
                continue

            # normalize points
            target_partition_points = np.copy(self.separated_box_points[target_gt_idx][non_empty_partition_idx])
            target_gt_center, target_gt_dim, target_gt_angle = self.gt_boxes[target_gt_idx][:3], self.gt_boxes[target_gt_idx][3:6], \
                                                               self.gt_boxes[target_gt_idx][6:7]
            target_partition_points[:, :3] -= target_gt_center
            target_partition_points[:, :3] = box_np_ops.rotation_3d_in_axis(target_partition_points[:, :3], -target_gt_angle, axis=2)
            target_partition_points[:, :3] /= target_gt_dim

            # swap points
            cur_gt_center, cur_gt_dim, cur_gt_angle = self.gt_boxes[i][:3], self.gt_boxes[i][3:6], self.gt_boxes[i][6:7]
            target_partition_points[:, :3] *= cur_gt_dim
            target_partition_points[:, :3] = box_np_ops.rotation_3d_in_axis(target_partition_points[:, :3],
                                                                            cur_gt_angle, axis=2)
            target_partition_points[:, :3] += cur_gt_center
            self.separated_box_points[i][non_empty_partition_idx] = target_partition_points
            self.aug_flag[i, non_empty_partition_idx, 1] = True

    def swap_partitions_random(self, num_swap=1, distance_limit=100, p=1.0, gt_box_idx=None):
        if num_swap <= 0:
            return

        for i in range(self.num_gt_boxes):
            if gt_box_idx is not None and i != gt_box_idx:
                continue
            if self.gt_boxes[i][0] > distance_limit:
                continue
            if np.random.rand(1) > p:
                continue
            gt_idxes = list(range(self.num_gt_boxes))

            if self.num_classes > 1:
                # remove idxes of different classes
                same_class_mask = self.gt_names == self.gt_names[i]
                same_class_mask[i] = False
                gt_idxes = [i for (i, m) in zip(gt_idxes, same_class_mask) if m]
            else:
                gt_idxes.remove(i)

            # find non-empty partition idx for both gt
            non_empty_partition_idx = -1

            if len(gt_idxes) > 0:
                target_gt_idx = np.random.choice(gt_idxes, 1, replace=False)[0]
                non_empty_partition_idxes = [idx for idx, points in enumerate(self.separated_box_points[i]) if points.shape[0] != 0]
                if len(non_empty_partition_idxes) == 0:
                    continue
                non_empty_partition_idx = np.random.choice(non_empty_partition_idxes, 1, replace=False)[0]
                target_non_empty_partition_idxes = [idx for idx, points in enumerate(self.separated_box_points[target_gt_idx]) if points.shape[0] != 0]
                if len(target_non_empty_partition_idxes) == 0:
                    continue
                target_non_empty_partition_idx = np.random.choice(target_non_empty_partition_idxes, 1, replace=False)[0]

            if non_empty_partition_idx == -1:
                continue

            # normalize points
            target_partition_points = np.copy(self.separated_box_points[target_gt_idx][target_non_empty_partition_idx])
            target_partition_boxes = np.array(self.random_partition_boxes[target_gt_idx][target_non_empty_partition_idx])
            target_gt_center, target_gt_dim, target_gt_angle = target_partition_boxes[:3], target_partition_boxes[3:6], \
                                                               target_partition_boxes[6:7]
            target_partition_points[:, :3] -= target_gt_center
            target_partition_points[:, :3] = box_np_ops.rotation_3d_in_axis(target_partition_points[:, :3], -target_gt_angle, axis=2)
            target_partition_points[:, :3] /= target_gt_dim

            # swap points
            cur_partition_boxes = np.array(self.random_partition_boxes[i][non_empty_partition_idx])
            cur_gt_center, cur_gt_dim, cur_gt_angle = cur_partition_boxes[:3], cur_partition_boxes[3:6], cur_partition_boxes[6:7]
            target_partition_points[:, :3] *= cur_gt_dim
            target_partition_points[:, :3] = box_np_ops.rotation_3d_in_axis(target_partition_points[:, :3],
                                                                            cur_gt_angle, axis=2)
            target_partition_points[:, :3] += cur_gt_center
            self.separated_box_points[i][non_empty_partition_idx] = target_partition_points
            self.aug_flag[i, non_empty_partition_idx, 1] = True

    #######################################
    # mix points in non-empty partitions #
    #######################################
    def mix_partitions(self, num_mix=1, distance_limit=100, p=1.0, gt_box_idx=None):
        if num_mix <= 0:
            return

        for i in range(self.num_gt_boxes):
            if gt_box_idx is not None and i != gt_box_idx:
                continue
            if self.gt_boxes[i][0] > distance_limit:
                continue
            if np.random.rand(1) > p:
                continue
            gt_idxes = list(range(self.num_gt_boxes))
            if self.num_classes > 1:
                # remove idxes of different classes
                same_class_mask = self.gt_names == self.gt_names[i]
                same_class_mask[i] = False
                gt_idxes = [i for (i, m) in zip(gt_idxes, same_class_mask) if m]
            else:
                gt_idxes.remove(i)

            # find non-empty partition idx for both gt
            non_empty_partition_idx = -1
            while len(gt_idxes) > 0:
                target_gt_idx = np.random.choice(gt_idxes, 1, replace=False)[0]
                non_empty_partition_idxes = [idx for idx, points in enumerate(self.separated_box_points[i]) if
                                             points.shape[0] != 0]
                while len(non_empty_partition_idxes) > 0:
                    non_empty_partition_idx_candidate = \
                    np.random.choice(non_empty_partition_idxes, 1, replace=False)[0]
                    if self.separated_box_points[target_gt_idx][non_empty_partition_idx_candidate].shape[0] != 0:
                        non_empty_partition_idx = non_empty_partition_idx_candidate
                        break
                    non_empty_partition_idxes.remove(non_empty_partition_idx_candidate)
                if non_empty_partition_idx != -1:
                    break
                gt_idxes.remove(target_gt_idx)

            if non_empty_partition_idx == -1:
                continue

            # normalize points
            target_partition_points = np.copy(self.separated_box_points[target_gt_idx][non_empty_partition_idx])
            target_gt_center, target_gt_dim, target_gt_angle = self.gt_boxes[target_gt_idx][:3], self.gt_boxes[
                                                                                                     target_gt_idx][
                                                                                                 3:6], \
                                                               self.gt_boxes[target_gt_idx][6:7]
            target_partition_points[:, :3] -= target_gt_center
            target_partition_points[:, :3] = box_np_ops.rotation_3d_in_axis(target_partition_points[:, :3],
                                                                            -target_gt_angle, axis=2)
            target_partition_points[:, :3] /= target_gt_dim

            # mix points
            cur_gt_center, cur_gt_dim, cur_gt_angle = self.gt_boxes[i][:3], self.gt_boxes[i][3:6], self.gt_boxes[i][
                                                                                                   6:7]
            target_partition_points[:, :3] *= cur_gt_dim
            target_partition_points[:, :3] = box_np_ops.rotation_3d_in_axis(target_partition_points[:, :3],
                                                                            cur_gt_angle, axis=2)
            target_partition_points[:, :3] += cur_gt_center
            self.separated_box_points[i][non_empty_partition_idx] = np.concatenate(
                (self.separated_box_points[i][non_empty_partition_idx], target_partition_points), axis=0)
            self.aug_flag[i, non_empty_partition_idx, 2] = True

    def mix_partitions_random(self, num_mix=1, distance_limit=100, p=1.0, gt_box_idx=None):
        if num_mix <= 0:
            return

        for i in range(self.num_gt_boxes):
            if gt_box_idx is not None and i != gt_box_idx:
                continue
            if self.gt_boxes[i][0] > distance_limit:
                continue
            if np.random.rand(1) > p:
                continue
            gt_idxes = list(range(self.num_gt_boxes))
            if self.num_classes > 1:
                # remove idxes of different classes
                same_class_mask = self.gt_names == self.gt_names[i]
                same_class_mask[i] = False
                gt_idxes = [i for (i, m) in zip(gt_idxes, same_class_mask) if m]
            else:
                gt_idxes.remove(i)

            # find non-empty partition idx for both gt
            non_empty_partition_idx = -1

            if len(gt_idxes) > 0:
                target_gt_idx = np.random.choice(gt_idxes, 1, replace=False)[0]
                non_empty_partition_idxes = [idx for idx, points in enumerate(self.separated_box_points[i]) if
                                             points.shape[0] != 0]
                if len(non_empty_partition_idxes) == 0:
                    continue
                non_empty_partition_idx = np.random.choice(non_empty_partition_idxes, 1, replace=False)[0]
                target_non_empty_partition_idxes = [idx for idx, points in
                                                    enumerate(self.separated_box_points[target_gt_idx]) if
                                                    points.shape[0] != 0]
                if len(target_non_empty_partition_idxes) == 0:
                    continue
                target_non_empty_partition_idx = \
                np.random.choice(target_non_empty_partition_idxes, 1, replace=False)[0]

            if non_empty_partition_idx == -1:
                continue

            # normalize points
            target_partition_points = np.copy(self.separated_box_points[target_gt_idx][target_non_empty_partition_idx])
            target_partition_boxes = np.array(
                self.random_partition_boxes[target_gt_idx][target_non_empty_partition_idx])
            target_gt_center, target_gt_dim, target_gt_angle = target_partition_boxes[:3], target_partition_boxes[3:6], \
                                                               target_partition_boxes[6:7]
            target_partition_points[:, :3] -= target_gt_center
            target_partition_points[:, :3] = box_np_ops.rotation_3d_in_axis(target_partition_points[:, :3],
                                                                            -target_gt_angle, axis=2)
            target_partition_points[:, :3] /= target_gt_dim

            # mix points
            cur_partition_boxes = np.array(self.random_partition_boxes[i][non_empty_partition_idx])
            cur_gt_center, cur_gt_dim, cur_gt_angle = cur_partition_boxes[:3], cur_partition_boxes[3:6], cur_partition_boxes[6:7]
            target_partition_points[:, :3] *= cur_gt_dim
            target_partition_points[:, :3] = box_np_ops.rotation_3d_in_axis(target_partition_points[:, :3],
                                                                            cur_gt_angle, axis=2)
            target_partition_points[:, :3] += cur_gt_center
            self.separated_box_points[i][non_empty_partition_idx] = np.concatenate(
                (self.separated_box_points[i][non_empty_partition_idx], target_partition_points), axis=0)
            self.aug_flag[i, non_empty_partition_idx, 2] = True

    #############################
    # dense partition to sparse #
    #############################
    def make_points_sparse(self, num_points_limit=1000, distance_limit=100, p=1.0, FPS=False, gt_box_idx=None):
        if num_points_limit <= 0:
            return

        for i in range(self.num_gt_boxes):
            if gt_box_idx is not None and i != gt_box_idx:
                continue
            if self.gt_boxes[i][0] > distance_limit:
                continue
            for j in range(self.num_partition[self.gt_names[i]]):
                if self.separated_box_points[i][j].shape[0] > num_points_limit:
                    if np.random.rand(1) > p:
                        continue
                    if FPS:
                        sparse_points_idx = farthest_point_sampling(self.separated_box_points[i][j][:, :3], num_points_limit)
                    else:
                        sparse_points_idx = np.random.choice(range(self.separated_box_points[i][j].shape[0]), num_points_limit, replace=False)
                    self.separated_box_points[i][j] = self.separated_box_points[i][j][sparse_points_idx]
                    self.aug_flag[i, j, 3] = True

    ########################################
    # translate points with gaussian noise #
    ########################################
    def jittering(self, sigma=0.01, p=0.5, distance_limit=100, gt_box_idx=None):
        for i in range(self.num_gt_boxes):
            if gt_box_idx is not None and i != gt_box_idx:
                continue
            if self.gt_boxes[i][0] > distance_limit:
                continue
            for j in range(self.num_partition[self.gt_names[i]]):
                if self.separated_box_points[i][j].shape[0] <= 0:
                    continue

                if np.random.rand(1) > p:
                    continue

                translation_noise = np.random.normal(0, sigma, size=self.separated_box_points[i][j].shape)

                self.separated_box_points[i][j] += translation_noise
                self.aug_flag[i, j, 4] = True


    ###################################################
    # generate random points in a specified partition #
    ###################################################
    def generate_random_noise(self, num_points=30, distance_limit=100, p=0.5, gt_box_idx=None):
        if num_points <= 0:
            return

        for i in range(self.num_gt_boxes):
            if gt_box_idx is not None and i != gt_box_idx:
                continue
            if self.gt_boxes[i][0] > distance_limit:
                continue

            for j in range(self.num_partition[self.gt_names[i]]):
                if np.random.rand(1) > p:
                    continue
                center = np.expand_dims(self.gt_boxes[i][:3], axis=0)
                dim = np.expand_dims(self.gt_boxes[i][3:6], axis=0)
                angle = self.gt_boxes[i][6:7]

                corners = np.squeeze(box_np_ops.corners_nd(dim, origin=(0.5, 0.5, 0.5)), axis=0)
                augmented_box_corners = augment_box_corners(corners)
                partition_corners = get_partition_corners(augmented_box_corners, j, self.gt_names[i])

                x_min, x_max = partition_corners[0, :, 0].min(), partition_corners[0, :, 0].max()
                y_min, y_max = partition_corners[0, :, 1].min(), partition_corners[0, :, 1].max()
                z_min, z_max = partition_corners[0, :, 2].min(), partition_corners[0, :, 2].max()

                generated_points = np.zeros((1, num_points, 4))
                generated_points[0, :, 0] = np.random.uniform(low=x_min, high=x_max, size=(num_points,))
                generated_points[0, :, 1] = np.random.uniform(low=y_min, high=y_max, size=(num_points,))
                generated_points[0, :, 2] = np.random.uniform(low=z_min, high=z_max, size=(num_points,))
                generated_points[0, :, 3] = np.random.uniform(low=0.0, high=1.0, size=(num_points,))

                generated_points[:, :, :3] = box_np_ops.rotation_3d_in_axis(generated_points[:, :, :3], angle, axis=2)
                generated_points[:, :, :3] += center.reshape([-1, 1, 3])
                self.separated_box_points[i][j] = np.concatenate((self.separated_box_points[i][j], generated_points[0]), axis=0)
                self.aug_flag[i, j, 5] = True
                

    def generate_random_noise_random(self, num_points=30, distance_limit=100, p=0.5, gt_box_idx=None):
        if num_points <= 0:
            return

        for i in range(self.num_gt_boxes):
            if gt_box_idx is not None and i != gt_box_idx:
                continue
            if self.gt_boxes[i][0] > distance_limit:
                continue

            for j in range(self.num_partition[self.gt_names[i]]):
                if np.random.rand(1) > p:
                    continue
                center = np.expand_dims(np.array(self.random_partition_boxes[i][j][:3]), axis=0)
                dim = np.expand_dims(np.array(self.random_partition_boxes[i][j][3:6]), axis=0)
                angle = np.array(self.random_partition_boxes[i][j][6:7])

                corners = np.squeeze(box_np_ops.corners_nd(dim, origin=(0.5, 0.5, 0.5)), axis=0)

                x_min, x_max = corners[:, 0].min(), corners[:, 0].max()
                y_min, y_max = corners[:, 1].min(), corners[:, 1].max()
                z_min, z_max = corners[:, 2].min(), corners[:, 2].max()

                generated_points = np.zeros((1, num_points, 4))
                generated_points[0, :, 0] = np.random.uniform(low=x_min, high=x_max, size=(num_points,))
                generated_points[0, :, 1] = np.random.uniform(low=y_min, high=y_max, size=(num_points,))
                generated_points[0, :, 2] = np.random.uniform(low=z_min, high=z_max, size=(num_points,))
                generated_points[0, :, 3] = np.random.uniform(low=0.0, high=1.0, size=(num_points,))

                generated_points[:, :, :3] = box_np_ops.rotation_3d_in_axis(generated_points[:, :, :3], angle, axis=2)
                generated_points[:, :, :3] += center.reshape([-1, 1, 3])
                self.separated_box_points[i][j] = np.concatenate((self.separated_box_points[i][j], generated_points[0]), axis=0)
                self.aug_flag[i, j, 5] = True



    ###############################
    # apply methods independently #
    ###############################
    def augment(self, pa_aug_param):
        pa_aug_param = self.interpret_pa_aug_param(pa_aug_param)
        self.dropout_partitions(num_dropout_partition=pa_aug_param['dropout'], p=pa_aug_param['dropout_p'], distance_limit=pa_aug_param['distance'])
        if self.random_partition:
            self.swap_partitions_random(num_swap=pa_aug_param['swap'], p=pa_aug_param['swap_p'], distance_limit=pa_aug_param['distance'])
            self.mix_partitions_random(num_mix=pa_aug_param['mix'], p=pa_aug_param['mix_p'], distance_limit=pa_aug_param['distance'])
        else:
            self.swap_partitions(num_swap=pa_aug_param['swap'], p=pa_aug_param['swap_p'], distance_limit=pa_aug_param['distance'])
            self.mix_partitions(num_mix=pa_aug_param['mix'], p=pa_aug_param['mix_p'], distance_limit=pa_aug_param['distance'])
        self.make_points_sparse(num_points_limit=pa_aug_param['sparse'], p=pa_aug_param['sparse_p'], FPS=True, distance_limit=pa_aug_param['distance'])
        self.jittering(sigma=pa_aug_param['jitter'], p=pa_aug_param['jitter_p'], distance_limit=pa_aug_param['distance'])
        if self.random_partition:
            self.generate_random_noise_random(num_points=pa_aug_param['noise'], p=pa_aug_param['noise_p'], distance_limit=pa_aug_param['distance'])
        else:
            self.generate_random_noise(num_points=pa_aug_param['noise'], p=pa_aug_param['noise_p'], distance_limit=pa_aug_param['distance'])

        fg_points = self.stack_fg_points()
        self.points = np.vstack((fg_points, self.bg_points))
        return self.points, self.gt_boxes_mask



    def generate_noise_robustness_test(self, noise_ratio=0.1, remove_original_points=False):
        num_points = self.points.shape[0]
        x_min, x_max = self.points[:, 0].min(), self.points[:, 0].max()
        y_min, y_max = self.points[:, 1].min(), self.points[:, 1].max()
        z_min, z_max = self.points[:, 2].min(), self.points[:, 2].max()
        r_min, r_max = self.points[:, 3].min(), self.points[:, 3].max()
        false_idx = np.random.choice(range(num_points), int(num_points * noise_ratio), replace=False)
        mask = np.ones(num_points, dtype=np.bool)
        mask[false_idx] = False
        self.points = self.points[mask]

        num_noise = int(num_points * noise_ratio)
        generated_points = np.zeros((num_noise, 4))
        generated_points[:, 0] = np.random.uniform(low=x_min, high=x_max, size=(num_noise,))
        generated_points[:, 1] = np.random.uniform(low=y_min, high=y_max, size=(num_noise,))
        generated_points[:, 2] = np.random.uniform(low=z_min, high=z_max, size=(num_noise,))
        generated_points[:, 3] = np.random.uniform(low=r_min, high=r_max, size=(num_noise,))

        self.points = np.concatenate((self.points, generated_points), axis=0)

    def sparse_robustness_test(self, sparse_ratio=0.8):
        num_points = self.points.shape[0]
        num_points_limit = int(num_points * sparse_ratio)
        sparse_points_idx = farthest_point_sampling(self.points[:, :3], num_points_limit)
        self.points = self.points[sparse_points_idx]

    def jitter_robustness_test(self, sigma=0.01):
        num_points = self.points.shape[0]
        translation_noise = np.random.normal(0, sigma, size=[num_points, 3])
        self.points[:, :3] += translation_noise



    def create_robusteness_test_data(self, test_name="KITTI-D"):
        if test_name == "KITTI-D":
            self.dropout_partitions(num_dropout_partition=1, p=1.0, robustness_test=True)
            fg_points = self.stack_fg_points()
            self.points = np.vstack((fg_points, self.bg_points))
        elif test_name == "KITTI-N":
            self.generate_noise_robustness_test(noise_ratio=0.2, remove_original_points=True)
        elif test_name == "KITTI-S":
            self.sparse_robustness_test(sparse_ratio=0.3)
        elif test_name == "KITTI-J":
            self.jitter_robustness_test(sigma=0.1)
        else:
            print()
            
        return self.points, self.gt_boxes_mask, self.aug_flag, self.partition_corners

