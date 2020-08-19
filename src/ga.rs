use rand::prelude::*;
use std::cmp;

use crate::map::{Map, Path};

pub struct Settings {
    pub replace_percent: f32,
    pub elitist_percent: f32,
    pub crossover_prob: f32,
    pub mutate_prob: f32
}

pub trait Chromosome: Path + Clone {
    fn len(&self) -> usize;

    fn crossover(&mut self, other: &mut Self, start: usize, end: usize);

    fn point_mutation(&mut self, index: usize, rng: &mut impl Rng);
}

#[inline]
pub fn slice_crossover<T>(first: &mut [T], second: &mut [T], start: usize, end: usize) {
    (&mut first[start..end]).swap_with_slice(&mut second[start..end]);
}

pub trait Recombinator {
    fn recombine<C: Chromosome>(&self, first: &mut C, second: &mut C, rng: &mut impl Rng);
}

pub struct KPoint {
    count: f32
}

impl KPoint {
    pub fn new(count: usize) -> Self {
        assert!(count > 0);

        KPoint {
            count: (count - 1) as f32
        }
    }

    fn single_point<C: Chromosome>(first: &mut C, second: &mut C, rng: &mut impl Rng) {
        let cut = (rng.gen::<f32>() * first.len() as f32) as usize;
        first.crossover(second, cut, first.len());
    }
}

impl Recombinator for KPoint {
    fn recombine<C: Chromosome>(&self, first: &mut C, second: &mut C, rng: &mut impl Rng) {
        assert_eq!(first.len(), second.len(), "Cannot recombine chromosomes of different lengths");

        if self.count == 0.0 {
            Self::single_point(first, second, rng);
            return;
        }

        let step = (first.len() as f32) / (self.count + 1.0);
        let mut i = 0.0f32;
        let mut start: f32;
        let mut end = rng.gen::<f32>() * step;
        while i < self.count {
            i += 1.0;
            start = end;
            end = step * (rng.gen::<f32>() + i);
            first.crossover(second, start as usize, end.round() as usize);
        }
    }
}

pub struct Uniform {
    weight: f32
}

impl Uniform {
    pub fn new() -> Self {
        Uniform {
            weight: 0.5
        }
    }

    pub fn weighted(weight: f32) -> Self {
        assert!(weight > 0.0 && weight <= 0.5, "Weight must be on the interval (0.0, 0.5]");

        Uniform {
            weight
        }
    }
}

impl Recombinator for Uniform {
    fn recombine<C: Chromosome>(&self, first: &mut C, second: &mut C, rng: &mut impl Rng) {
        assert_eq!(first.len(), second.len(), "Cannot recombine chromosomes of different lengths");

        let mut copy: bool = rng.gen();
        let mut last_index = 0usize;
        for i in 1..first.len() {
            if rng.gen::<f32>() > self.weight {
                continue;
            }

            if copy {
                first.crossover(second, last_index, i);
            }

            copy = !copy;
            last_index = i;
        }

        if copy {
            first.crossover(second, last_index, first.len());
        }
    }
}

pub struct RouletteWheelSelection;

impl RouletteWheelSelection {
    pub fn evolve<C, R>(settings: &Settings, map: &Map, population: &mut Vec<C>, recombinator: &R, fix: bool) -> f32
    where
        C: Chromosome,
        R: Recombinator
    {
        let mut rng = thread_rng();

        // Compute the loss vector
        let n = population.len();
        let mut losses = vec![0.0f32; n];
        let mut loss_sum: f32 = 0.0;
        let mut min_loss = f32::MAX;
        for i in 0..n {
            let loss = population[i].evaluate(map);
            losses[i] = loss;
            loss_sum += loss;

            if loss < min_loss {
                min_loss = loss;
            }
        }
        losses.iter_mut().for_each(|loss| *loss /= loss_sum);

        // Get elitism cut-off
        if settings.elitist_percent > 0.0 {
            losses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(cmp::Ordering::Equal));
            population.sort_by(|a, b| {
                a.evaluate(map).partial_cmp(&b.evaluate(map)).unwrap_or(cmp::Ordering::Equal)
            });
        }

        let mut offspring_count = 0;
        let target_offspring_count = 1.max(((settings.replace_percent - settings.elitist_percent) * (n as f32)) as usize);
        while offspring_count < target_offspring_count {
            // Get the two parents
            let mut selections = [0usize; 2];
            'selector: for i in 0..2 {
                let mut random = rng.gen::<f32>();
                for j in 0..n {
                    if random < losses[j] && (i == 0 || selections[0] != j) {
                        selections[i] = j;
                        continue 'selector;
                    }

                    random -= losses[j];
                }

                selections[i] = n - 1;
            }

            // Compute the child chromosomes
            let mut first = population[selections[0]].clone();
            let mut second = population[selections[1]].clone();
            if rng.gen::<f32>() < settings.crossover_prob {
                recombinator.recombine(&mut first, &mut second, &mut rng);
            }
            if rng.gen::<f32>() < settings.mutate_prob {
                first.point_mutation(rng.gen::<usize>() % first.len(), &mut rng);
            }
            if rng.gen::<f32>() < settings.mutate_prob {
                second.point_mutation(rng.gen::<usize>() % second.len(), &mut rng);
            }

            // Fix trivial errors
            if fix {
                first.fix(&map);
                second.fix(&map);
            }

            // Update minimum loss value
            let loss = first.evaluate(map);
            if loss < min_loss {
                min_loss = loss;
            }
            let loss = second.evaluate(map);
            if loss < min_loss {
                min_loss = loss;
            }
            
            // Add them to the population
            population.push(first);
            offspring_count += 1;

            if offspring_count < target_offspring_count {
                population.push(second);
                offspring_count += 1;
            }
        }

        let mut index = n - 1;
        while population.len() > n {
            population.remove(index);
            index -= 1;
        }

        min_loss
    }
}