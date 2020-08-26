use crate::map::{Map, Path, Point};
use std::cell::Cell;
use std::cmp;
use std::fmt::{self, Debug, Formatter};
use std::ops::Deref;

pub struct Counter<'a>(Option<&'a mut usize>);

impl<'a> Counter<'a> {
    pub fn increment(&mut self) {
        if let Some(inner) = self.0.as_deref_mut() {
            *inner += 1;
        }
    }
}

impl<'a> From<Option<&'a mut usize>> for Counter<'a> {
    fn from(x: Option<&'a mut usize>) -> Self {
        Counter(x)
    }
}

impl<'a> From<&'a mut usize> for Counter<'a> {
    fn from(x: &'a mut usize) -> Self {
        Counter(Some(x))
    }
}

pub fn brute_force<'a, C: Into<Counter<'a>>>(map: &Map, counter: C) -> (Vec<usize>, f32) {
    let mut counter = counter.into();
    let mut current = vec![0usize; map.size()];
    current.iter_mut().enumerate().for_each(|(index, ele)| *ele = index);
    let mut solution = current.clone();
    let mut shortest_dist = current.evaluate(map);

    let max = map.size() - 1;
    let mut increase: usize = 0;
    while increase != max {
        counter.increment();

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

pub fn branch_and_bound<'a, C: Into<Counter<'a>>>(map: &Map, counter: C) -> f32 {
    let mut counter = counter.into();
    let data = PathData::new(map);
    let mut min_dist = nearest_neighbor(map);

    let mut iter = data.iter();
    while let Some(point) = iter.next() {
        branch_and_bound_internal(data.iter(), &point, 0f32, &mut min_dist, &mut counter);

        // Explicit for clarity
        drop(point);
    } 

    min_dist
}


fn branch_and_bound_internal(
    mut points: PathDataIter<'_>,
    last: &Point,
    accumulated: f32,
    min_dist: &mut f32,
    counter: &mut Counter<'_>
) {
    let mut count = 0;
    while let Some(point) = points.next() {
        count += 1;
        counter.increment();
        
        let new_accumulated = accumulated + point.dist(last);
        if points.lower_bound(new_accumulated) < *min_dist {
            branch_and_bound_internal(points.clone_reset(), &point, new_accumulated, min_dist, counter);
        }

        // Explicit for clarity
        drop(point);
    }

    if count == 0 && accumulated < *min_dist {
        *min_dist = accumulated;
    }
}

struct PathData {
    points: Box<[(Point, f32)]>,
    visited: Box<[Cell<bool>]>
}

impl PathData {
    fn new(map: &Map) -> Self {
        // Get a list of the points, each with its distance to its nearest neighbor
        let mut points = vec![(Point::new(), 0f32); map.size()].into_boxed_slice();
        for i in 0..map.size() {
            let point = map.get(i).unwrap().clone();
            let mut min = f32::MAX;
            for j in 0..map.size() {
                let dist = map.get(j).map(|p| p.dist(&point)).unwrap_or(f32::MAX);
                if i != j && dist < min {
                    min = dist;
                }
            }
            points[i] = (point, min);
        }

        PathData {
            points,
            visited: vec![Cell::new(false); map.size()].into_boxed_slice()
        }
    }

    const fn iter(&self) -> PathDataIter<'_> {
        PathDataIter::new(self)
    }

    #[inline]
    fn lower_bound(&self, accumulated: f32) -> f32 {
        // The lower bound is calculated by summing the remaining nearest-neighbor distances (excluding one)
        // and adding that to the current accumulated distance.

        accumulated + self.visited.iter()
            .enumerate()
            .filter(|(_, flag)| !flag.get())
            .skip(1)
            // Infallible: visited.len() == points.len()
            .map(|(index, _)| unsafe { self.points.get_unchecked(index).1 })
            .sum::<f32>()
    }

    #[inline]
    fn visit_next(&self, index: &mut usize) -> Option<VisitedPoint<'_>> {
        while self.visited.get(*index).map(Cell::get).unwrap_or(false) {
            *index += 1;
        }

        if *index == self.points.len() {
            return None;
        }

        // Index checked beforehand
        unsafe  {
            self.visited.get_unchecked(*index).set(true);

            let ret = Some(VisitedPoint{
                value: self.points.get_unchecked(*index).0.clone(),
                source: self,
                index: *index
            });

            *index += 1;
            ret
        }
    }
}

struct PathDataIter<'a> {
    path_data: &'a PathData,
    index: usize
}

impl<'a> PathDataIter<'a> {
    const fn new(path_data: &'a PathData) -> Self {
        PathDataIter {
            path_data,
            index: 0
        }
    }

    #[inline]
    fn lower_bound(&self, accumulated: f32) -> f32 {
        self.path_data.lower_bound(accumulated)
    }

    const fn clone_reset(&self) -> Self {
        Self::new(self.path_data)
    }
}

impl<'a> Iterator for PathDataIter<'a> {
    type Item = VisitedPoint<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.path_data.visit_next(&mut self.index)
    }
}

struct VisitedPoint<'a> {
    value: Point,
    source: &'a PathData,
    index: usize
}

impl<'a> Deref for VisitedPoint<'a> {
    type Target = Point;

    #[inline]
    fn deref(&self) -> &Point {
        &self.value
    }
}

impl<'a> Drop for VisitedPoint<'a> {
    #[inline]
    fn drop(&mut self) {
        // Infallible: in order for this type to be constructed the index must be valid
        unsafe {
            self.source.visited.get_unchecked(self.index).set(false);
        }
    }
}

impl<'a> Debug for VisitedPoint<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.value, f)
    }
}