// use scalar::Scalar;
// #[derive(Scalar, Debug)]
// pub struct TrivialField(u64);

// pub trait TrivialFieldTrait{
//   fn one()->Self;
//   fn zero()->Self;
//   fn to_bytes_le(&self) -> [u8;32];
//   fn neg(&mut self);
// }

// impl TrivialFieldTrait for TrivialField{
//   fn one()-> Self{
//     TrivialField(1)
//   }
//   fn zero()-> Self{
//     TrivialField(0)
//   }

//   fn to_bytes_le(&self) -> [u8;32]{
//     let tmp = self.0.to_le_bytes();
//     let mut expanded:[u8;32] = [0;32];
//     expanded.copy_from_slice(&tmp);
//     expanded
//   }

//   fn neg(&mut self){

//   }
// }