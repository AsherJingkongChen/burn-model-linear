use burn::{
    backend,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{activation, backend::Backend, Distribution, Tensor},
};
use kdam::{tqdm, BarExt};

fn main() -> anyhow::Result<()> {
    type Backend = backend::NdArray;
    Backend::seed(1);

    const ITERS: usize = 100000;

    let model = Model::<Backend>::init(&Default::default());

    let input =
        Tensor::<Backend, 1>::random([256], Distribution::Normal(0.0, 1.0), &Default::default())
            .set_require_grad(false);
    let target =
        Tensor::<Backend, 1>::random([2], Distribution::Normal(0.0, 1.0), &Default::default())
            .set_require_grad(false);

    let mut bar = tqdm!(total = ITERS);
    for _ in 0..ITERS {
        let output = model.forward(input.clone());
        let _ = get_loss_mse(output, target.clone());

        let _ = bar.update(1);
    }

    Ok(())
}

#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    pub l1: Linear<B>,
    pub l2: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn init(device: &B::Device) -> Self {
        Self {
            l1: LinearConfig::new(256, 256).init(device),
            l2: LinearConfig::new(256, 2).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        self.l2.forward(activation::sigmoid(self.l1.forward(input)))
    }
}

pub fn get_loss_mse<B: Backend>(output: Tensor<B, 1>, target: Tensor<B, 1>) -> Tensor<B, 1> {
    output.sub(target).powf_scalar(2.0).mean()
}
