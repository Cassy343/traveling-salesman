#![allow(dead_code, unused_imports)]

mod ga;
mod map;
mod solve;

use ga::*;
use map::{
    Map,
    Path,
    Point,
    RandomKeyPath,
    RemovalIndex,
    SwapPath
};
use solve::{brute_force, nearest_neighbor};
use std::f32::consts;
use std::fmt::Debug;
use std::time::SystemTime;

const SETTINGS: Settings = Settings {
    replace_percent: 1.0,
    elitist_percent: 0.25,
    crossover_prob: 0.9,
    mutate_prob: 0.05,
    selection_noise: 0.0
};

fn main() {
    let mut points = Vec::new();
    let n = 75;
    for i in 0..n {
        let theta = (i as f32 / n as f32) * 2.0 * consts::PI;
        points.push(Point::polar(1.0, theta));
    }

    let map = Map::from_points(points);
    let mut population: Vec<RandomKeyPath> = Vec::with_capacity(5);
    population.resize_with(population.capacity(), || RandomKeyPath::new(&map));
    let recomb = Uniform::new();
    println!("Iters: {}", run(&map, nearest_neighbor(&map), population, &recomb, true));
}

fn average(iters: u32) -> (u32, u32) {
    let recomb = Uniform::new();

    let mut total_with_fix = 0u128;
    let mut total_without_fix = 0u128;
    for i in 0..iters {
        let map = Map::new(10);
        let target = brute_force(&map).1;

        let mut population: Vec<RandomKeyPath> = Vec::with_capacity(50);
        population.resize_with(population.capacity(), || RandomKeyPath::new(&map));

        total_with_fix += run(&map, target, population.clone(), &recomb, true) as u128;
        total_without_fix += run(&map, target, population.clone(), &recomb, false) as u128;
        println!("{}/{}...", i + 1, iters);
    }
    ((total_with_fix / iters as u128) as u32, (total_without_fix / iters as u128) as u32)
}

fn run<C: Chromosome + Debug, R: Recombinator>(map: &Map, target: f32, mut population: Vec<C>, recomb: &R, fix: bool) -> u32 {
    let mut min_loss = f32::MAX;
    let mut iterations = 0u32;
    const MAX_ITERATIONS: u32 = 1_000_000;
    while min_loss - target > 1e-5 && iterations < MAX_ITERATIONS {
        let loss = RouletteWheelSelection::evolve(&SETTINGS, &map, &mut population, recomb, fix);
        if loss < min_loss {
            min_loss = loss;
            println!("{}, {}, {}", target, loss, iterations);
        }

        iterations += 1;
    }

    if iterations == MAX_ITERATIONS {
        population.iter().for_each(|indv| println!("{:?}", indv));
    }

    iterations
}