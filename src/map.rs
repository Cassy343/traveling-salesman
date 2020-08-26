use itertools::Itertools;
use rand::prelude::*;
use std::cmp;
use std::convert::AsRef;
use std::f32::consts;
use std::fmt::{self, Debug, Display, Formatter};
use std::mem;
use std::ops::{Index, IndexMut};

use crate::ga::{Chromosome, slice_crossover};

#[derive(Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32
}

impl Point {
    pub fn new() -> Self {
        Point {
            x: 0.0,
            y: 0.0
        }
    }

    pub fn polar(radius: f32, theta: f32) -> Self {
        Point {
            x: radius * theta.cos(),
            y: radius * theta.sin()
        }
    }

    #[inline]
    pub fn dist(&self, other: &Point) -> f32 {
        let x = self.x - other.x;
        let y = self.y - other.y;
        f32::hypot(x, y)
    }

    #[inline]
    pub fn dist_sq(&self, other: &Point) -> f32 {
        let x = self.x - other.x;
        let y = self.y - other.y;
        x * x + y * y
    }
}

impl Display for Point {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl Debug for Point {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

#[derive(Clone)]
pub struct Map {
    points: Box<[Point]>
}

impl Map {
    pub fn new(count: usize) -> Self {
        let mut rng = thread_rng();
        let mut points = vec![Point::new(); count].into_boxed_slice();

        for i in 0..count {
            let theta = 2.0 * consts::PI * rng.gen::<f32>();
            let radius = rng.gen::<f32>();

            let point = &mut points[i];
            point.x = radius * theta.cos();
            point.y = radius * theta.sin();
        }

        Map {
            points
        }
    }

    pub fn from_points(points: Vec<Point>) -> Self {
        Map {
            points: points.into_boxed_slice()
        }
    }

    pub fn size(&self) -> usize {
        self.points.len()
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<Point> {
        self.points.get(index).cloned()
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut Point> {
        self.points.get_mut(index)
    }

    pub fn swap(&mut self, first: usize, second: usize) {
        let (first_mut, second_mut) = self.points.split_at_mut(second);
        mem::swap(&mut first_mut[first], &mut second_mut[0]);
    }

    pub fn clone_to_vec(&self) -> Vec<Point> {
        let mut result = Vec::with_capacity(self.points.len());
        result.extend_from_slice(&self.points);
        result
    }
}

impl Index<usize> for Map {
    type Output = Point;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.points[index]
    }
}

impl IndexMut<usize> for Map {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.points[index]
    }
}

impl Debug for Map {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.points, f)
    }
}

pub trait Path {
    fn evaluate(&self, map: &Map) -> f32;

    fn reorder(&self, map: &mut Map);

    fn fix(&mut self, _map: &Map) {}
}

impl<T: AsRef<[usize]>> Path for T {
    fn evaluate(&self, map: &Map) -> f32 {
        self.as_ref().windows(2)
            .flat_map(|segment| Some((map.get(segment[0])?, map.get(segment[1])?)))
            .map(|(a, b)| a.dist(&b))
            .sum::<f32>()
    }

    fn reorder(&self, map: &mut Map) {
        let path = self.as_ref();
        for i in 0..path.len() {
            map.swap(i, path[i]);
        }
    }
}

#[derive(Clone)]
pub struct RemovalIndex {
    path: Box<[usize]>
}

impl RemovalIndex {
    pub fn new(map: &Map) -> Self {
        let mut rng = thread_rng();
        let mut max = map.size();
        let mut path = vec![0usize; max - 1].into_boxed_slice();
        for index in path.iter_mut() {
            *index = rng.gen::<usize>() % max;
            max -= 1;
        }

        RemovalIndex {
            path
        }
    }

    pub fn in_order(map: &Map) -> Self {
        RemovalIndex {
            path: vec![0usize; map.size() - 1].into_boxed_slice()
        }
    }

    pub fn inner_mut(&mut self) -> &mut Box<[usize]> {
        &mut self.path
    }
}

impl Debug for RemovalIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.path, f)
    }
}

impl Path for RemovalIndex {
    fn evaluate(&self, map: &Map) -> f32 {
        if map.size() != self.path.len() + 1 {
            return 0.0;
        }

        let mut cloned = map.clone_to_vec();
        let mut last: Point = cloned.remove(self.path[0]);
        let mut total: f32 = 0.0;
        for i in 1..self.path.len() {
            let current = cloned.remove(self.path[i]);
            total += current.dist(&last);
            last = current;
        }

        total + last.dist(&cloned[0])
    }

    fn reorder(&self, map: &mut Map) {
        if map.size() != self.path.len() + 1 {
            return;
        }

        let mut cloned = map.clone_to_vec();
        for i in 0..self.path.len() {
            *map.get_mut(i).unwrap() = cloned.remove(self.path[i]);
        }
        *map.get_mut(map.size() - 1).unwrap() = cloned[0];
    }
}

impl Chromosome for RemovalIndex {
    fn len(&self) -> usize {
        self.path.len()
    }

    fn crossover(&mut self, other: &mut Self, start: usize, end: usize) {
        slice_crossover(&mut self.path, &mut other.path, start, end);
    }

    fn point_mutation(&mut self, index: usize, rng: &mut impl Rng) {
        self.path[index] = rng.gen::<usize>() % (self.path.len() - index);
    }
}

#[derive(Clone)]
pub struct RandomKeyPath {
    key: Box<[f32]>
}

impl RandomKeyPath {
    pub fn new(map: &Map) -> Self {
        let mut key = vec![0.0f32; map.size()].into_boxed_slice();
        let mut rng = thread_rng();
        key.iter_mut().for_each(|element| *element = rng.gen());

        RandomKeyPath {
            key
        }
    }

    fn as_index_path(&self) -> Vec<usize> {
        self.key.iter()
            .enumerate()
            .sorted_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or(cmp::Ordering::Equal)
            })
            .map(|(index, _)| index)
            .collect()
    }
}

impl Debug for RandomKeyPath {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.key, f)
    }
}

impl Path for RandomKeyPath {
    fn evaluate(&self, map: &Map) -> f32 {
        self.as_index_path().evaluate(map)
    }

    fn reorder(&self, map: &mut Map) {
        self.as_index_path().reorder(map)
    }

    fn fix(&mut self, map: &Map) {
        let len = self.key.len();
        if len < 3 {
            return;
        }

        let mut index_path: Vec<_> = self.key.iter_mut()
            .enumerate()
            .sorted_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or(cmp::Ordering::Equal)
            })
            .collect();

        // Fix the middle
        for i in 0..3 {
            let mut j = i;
            while j < index_path.len() - 3 {
                // Take a group of four and evaluate the current and alternative path lengths
                let section = &mut index_path[j..j + 4];
                let current = map[section[0].0].dist(&map[section[1].0]) + map[section[2].0].dist(&map[section[3].0]);
                let alternative = map[section[0].0].dist(&map[section[2].0]) + map[section[1].0].dist(&map[section[3].0]);

                // If the mid-swap improved the path length, apply it
                if alternative < current {
                    // Update the value in the key
                    let (section_first, section_second) = section.split_at_mut(2);
                    mem::swap(section_first[1].1, section_second[0].1);

                    // Update the copied path
                    let (path_first, path_second) = index_path.split_at_mut(j + 2);
                    mem::swap(&mut path_first[j + 1], &mut path_second[0]);
                }

                j += 3;
            }
        }

        // Fix the start point
        let anchor = &map[index_path[2].0];
        let current = map[index_path[1].0].dist_sq(anchor);
        let alternative = map[index_path[0].0].dist_sq(anchor);
        if alternative < current {
            let (first, second) = index_path.split_at_mut(1);
            mem::swap(first[0].1, second[0].1);
        }

        // Fix the end point
        if len > 5 {
            let anchor = &map[index_path[len - 3].0];
            let current = map[index_path[len - 2].0].dist_sq(anchor);
            let alternative = map[index_path[len - 1].0].dist_sq(anchor);
            if alternative < current {
                let (first, second) = index_path.split_at_mut(len - 1);
                mem::swap(first[len - 2].1, second[0].1);
            }
        }
    }
}

impl Chromosome for RandomKeyPath {
    fn len(&self) -> usize {
        self.key.len()
    }

    fn crossover(&mut self, other: &mut Self, start: usize, end: usize) {
        slice_crossover(&mut self.key, &mut other.key, start, end);
    }

    fn point_mutation(&mut self, index: usize, rng: &mut impl Rng) {
        self.key[index] = rng.gen();
    }
}

#[derive(Clone)]
pub struct SwapPath {
    swaps: Box<[usize]>,
    map_size: usize
}

impl SwapPath {
    pub fn new(map: &Map, swap_count: usize) -> Self {
        let map_size = map.size();
        let mut swaps = vec![0usize; swap_count * 2].into_boxed_slice();
        let mut rng = thread_rng();
        for i in (0..swap_count * 2).step_by(2) {
            swaps[i] = rng.gen::<usize>() % map_size;
            swaps[i + 1] = rng.gen::<usize>() % map_size;
        }

        SwapPath {
            swaps,
            map_size
        }
    }
}

impl Debug for SwapPath {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.swaps, f)
    }
}

impl Path for SwapPath {
    fn evaluate(&self, map: &Map) -> f32 {
        let mut cloned = map.clone_to_vec();
        for swap in self.swaps.chunks(2) {
            cloned.swap(swap[0], swap[1]);
        }

        cloned.windows(2).map(|points| points[0].dist(&points[1])).sum::<f32>()
    }

    fn reorder(&self, map: &mut Map) {
        for swap in self.swaps.chunks(2) {
            map.swap(swap[0], swap[1]);
        }
    }
}

impl Chromosome for SwapPath {
    fn len(&self) -> usize {
        self.swaps.len()
    }

    fn crossover(&mut self, other: &mut Self, start: usize, end: usize) {
        slice_crossover(&mut self.swaps, &mut other.swaps, start, end);
    }

    fn point_mutation(&mut self, index: usize, rng: &mut impl Rng) {
        self.swaps[index] = rng.gen::<usize>() % self.map_size;
    }
}