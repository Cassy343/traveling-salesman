use crate::map::{Map, Path};
use std::cmp;

pub fn brute_force(map: &Map) -> (Vec<usize>, f32) {
    let mut current = vec![0usize; map.size()];
    current.iter_mut().enumerate().for_each(|(index, ele)| *ele = index);
    let mut solution = current.clone();
    let mut shortest_dist = current.evaluate(map);

    let max = map.size() - 1;
    let mut increase: usize = 0;
    while increase != max {
        if increase == 0 {
            current.swap(increase, increase + 1);
            increase += 1;
            while increase < max && current[increase] > current[increase + 1] {
                increase += 1;
            }
        } else {
            if current[increase + 1] > current[0] {
                current.swap(increase + 1, 0);
            } else {
                let mut start = 0;
                let mut end = increase;
                let mut mid = (start + end) / 2;
                let top = current[increase + 1];
                while !(current[mid] < top && current[mid - 1] > top) {
                    if current[mid] < top {
                        end = mid - 1;
                    } else {
                        start = mid + 1;
                    }
                    mid = (start + end) / 2;
                }

                current.swap(increase + 1, mid);
            }

            for i in 0..=increase / 2 {
                current.swap(i, increase - i);
            }
            increase = 0;
        }

        let dist = current.evaluate(map);
        if dist < shortest_dist {
            shortest_dist = dist;
            solution = current.clone();
        }
    }

    (solution, shortest_dist)
}

pub fn nearest_neighbor(map: &Map) -> f32 {
    let mut points = map.clone_to_vec();
    let mut point = points.remove(0);
    let mut total: f32 = 0.0;
    while points.len() > 0 {
        let (index, nearest) = points.iter().enumerate().min_by(|(_, a), (_, b)| {
            let da = a.dist_sq(&point);
            let db = b.dist_sq(&point);
            da.partial_cmp(&db).unwrap_or(cmp::Ordering::Equal)
        }).unwrap();
        total += nearest.dist(&point);
        point = points.remove(index);
    }
    total
}