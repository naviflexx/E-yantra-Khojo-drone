#!/usr/bin/env python3

# =======================================================
# eYRC 2026-27: Khoj-o-Drone Task 1A Final Production
# Full HSV threshold arrays complete and fully syntactically valid.
# Exact literal text headers matched to instructions rules.
# =======================================================

import os
import argparse
import cv2
import numpy as np

def locate_arena_markers(src_frame):
    """Parses raw frame to grab ArUco marker boundaries."""
    gray_layer = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)
    
    if hasattr(cv2, 'aruco') and hasattr(cv2.aruco, 'DetectorParameters'):
        dictionary_preset = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        config_params = cv2.aruco.DetectorParameters()
        finder_engine = cv2.aruco.ArucoDetector(dictionary_preset, config_params)
        vertices, tag_ids, _ = finder_engine.detectMarkers(gray_layer)
    else:
        dictionary_preset = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
        config_params = cv2.aruco.DetectorParameters_create()
        vertices, tag_ids, _ = cv2.aruco.detectMarkers(gray_layer, dictionary_preset, parameters=config_params)
        
    extracted_tags = {}
    if tag_ids is not None:
        for idx, current_id in enumerate(tag_ids.flatten()):
            extracted_tags[int(current_id)] = vertices[idx]
            
    return extracted_tags

def warp_field_to_square(raw_img, tag_map):
    """Executes perspective correction to establish clean 900x900 playground matrix."""
    needed_tags = [80, 85, 90, 95]
    if any(tag not in tag_map for tag in needed_tags):
        return None

    # Retrieve mapping corners: TL(80), TR(85), BR(90), BL(95)
    pt_tl = tag_map[80][0][2]
    pt_tr = tag_map[85][0][3]
    pt_br = tag_map[90][0][0]
    pt_bl = tag_map[95][0][1]

    initial_plane = np.float32([pt_tl, pt_tr, pt_br, pt_bl])
    normalized_plane = np.float32([[0, 0], [900, 0], [900, 900], [0, 900]])

    warp_matrix = cv2.getPerspectiveTransform(initial_plane, normalized_plane)
    return cv2.warpPerspective(raw_img, warp_matrix, (900, 900))

def build_junction_matrix():
    """Generates localized lookup coordinate registry for nodes A1 through K11."""
    node_registry = {}
    alphabet_headers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    
    for x_step, col_letter in enumerate(alphabet_headers):
        pixel_x = int((x_step + 1) * 75)
        for row_num in range(1, 12):
            pixel_y = int(row_num * 75)
            node_registry[f"{col_letter}{row_num}"] = (pixel_x, pixel_y)
            
    return node_registry

def isolate_and_extract_targets(warped_canvas):
    """Separates mask channels via precise HSV threshold bounds and geometry checks."""
    hsv_space = cv2.cvtColor(warped_canvas, cv2.COLOR_BGR2HSV)
    
    # Precise Red boundaries (Dual range mapping with strict saturation floor)
    low_r1, high_r1 = np.array([0, 140, 60]), np.array([10, 255, 255])
    low_r2, high_r2 = np.array([170, 140, 60]), np.array([180, 255, 255])
    
    # Precise Yellow boundaries
    low_y, high_y   = np.array([20, 100, 100]), np.array([30, 255, 255])
    
    combined_red_mask = cv2.bitwise_or(cv2.inRange(hsv_space, low_r1, high_r1), cv2.inRange(hsv_space, low_r2, high_r2))
    yellow_mask = cv2.inRange(hsv_space, low_y, high_y)
    
    def resolve_centroids(binary_mask, is_red=False):
        shapes, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points_list = []
        for shape in shapes:
            if cv2.contourArea(shape) > 25:
                # Calculate the number of structural corners (vertices)
                perimeter = cv2.arcLength(shape, True)
                corners_approx = cv2.approxPolyDP(shape, 0.04 * perimeter, True)
                
                # Filter out linear grid lines / noise patterns
                if is_red and len(corners_approx) > 4:
                    continue  
                    
                geometric_moments = cv2.moments(shape)
                if geometric_moments["m00"] != 0:
                    # Enforce area minimum limits to discard smaller line fragments completely
                    min_pixel_area = 150 if is_red else 25
                    if cv2.contourArea(shape) > min_pixel_area:
                        target_x = int(geometric_moments["m10"] / geometric_moments["m00"])
                        target_y = int(geometric_moments["m01"] / geometric_moments["m00"])
                        points_list.append((target_x, target_y))
        return points_list

    return resolve_centroids(combined_red_mask, is_red=True), resolve_centroids(yellow_mask, is_red=False)

def assign_centers_to_nodes(detected_centers, node_matrix):
    """Computes nearest geometric neighbor node assignment for verified points."""
    assigned_nodes = []
    for px, py in detected_centers:
        shortest_distance = float('inf')
        matched_node = ""
        for node_id, (nx, ny) in node_matrix.items():
            linear_distance = np.hypot(px - nx, py - ny)
            if linear_distance < shortest_distance:
                shortest_distance = linear_distance
                matched_node = node_id
        if matched_node:
            assigned_nodes.append(matched_node)
    return sorted(list(set(assigned_nodes)))

def main():
    arg_engine = argparse.ArgumentParser()
    arg_engine.add_argument('--image', required=True, help="Target evaluation matrix")
    parsed_arguments = arg_engine.parse_args()

    target_path = parsed_arguments.image
    if not os.path.exists(target_path):
        return

    matrix_source = cv2.imread(target_path)
    active_markers = locate_arena_markers(matrix_source)
    ordered_ids = sorted([int(identity) for identity in active_markers.keys()])
    
    flat_view = warp_field_to_square(matrix_source, active_markers)
    if flat_view is None:
        return

    grid_system = build_junction_matrix()
    critical_spots, stable_spots = isolate_and_extract_targets(flat_view)
    
    final_critical = assign_centers_to_nodes(critical_spots, grid_system)
    final_stable = assign_centers_to_nodes(stable_spots, grid_system)

    folder_prefix = os.path.dirname(target_path)
    base_name = os.path.basename(target_path)
    pure_stem, _ = os.path.splitext(base_name)
    
    destination_file = os.path.join(folder_prefix, f"{pure_stem}_results.txt") if folder_prefix else f"{pure_stem}_results.txt"
    
    # Rules to the letter line formatting outputs [image_8pWkf_]
    with open(destination_file, 'w') as file_writer:
        file_writer.write(f"Detected marker IDs: {ordered_ids}\n")
        file_writer.write("\n")
        file_writer.write(f"Critical Survivors: {', '.join(final_critical)}\n")
        file_writer.write(f"Stable Survivors: {', '.join(final_stable)}\n")

if __name__ == '__main__':
    main()

