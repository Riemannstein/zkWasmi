use nalgebra::DMatrix;
use ark_bls12_381::Fq;
use ark_poly::polynomial::multivariate::{SparsePolynomial, SparseTerm};
// use ark_poly::MVPolynomial;
use ark_ff::fields::Field;
// use ark_ff::fields::BigInteger;
use crate::{core::UntypedValue, engine::Target};
use replace_with::replace_with_or_abort;
use ark_ff::bytes::ToBytes;
// use super::finite_field::{TrivialField, TrivialFieldTrait};


type GlobalStepCount = usize;
type VariableIndex = usize;
pub type VariableValue = Fq;

pub trait BigInteger{
  fn to_bytes_le(&self) -> [u8;32]{
    // TODO: Serialize
    [0;32]
  }
}

impl BigInteger for VariableValue{

}

#[derive(Debug)]
#[allow(non_snake_case)]
pub struct R1CS<T : Field + BigInteger>{
  variable_count : usize,
  global_step_count : GlobalStepCount,
  A : DMatrix<T>,
  B : DMatrix<T>,
  C:  DMatrix<T>,
  one : TraceVariable<T>,
  pc : TraceVariable<T>,
  // locals : Vec<TraceVariable<T>>,
  variables : Vec<TraceVariable<T>>,
  inputs : Vec<TraceVariable<T>>,
  // pub trace : BTreeMap<VariableIndex, TraceVariable>,
  // constraints : Vec<SparsePolynomial<VariableValue,SparseTerm>>
}

#[derive(Debug)]
pub enum TraceVariableKind
{
  PC,
  Other
}

#[derive(Debug)]
pub struct TraceVariable<T: Field+BigInteger>{
  pub kind : TraceVariableKind,
  pub global_step_count : GlobalStepCount,
  pub index : VariableIndex,
  pub value : T,
}

impl<T: Field + BigInteger> TraceVariable<T> {

  pub fn new(kind : TraceVariableKind, global_step_count : GlobalStepCount, index : Option<VariableIndex>, value : T) -> Self{
    let some_index : usize = match &kind {
      TraceVariableKind::PC => R1CS::<T>::INITIAL_VARIABLE_COUNT-1,
      TraceVariableKind::Other => index.unwrap()
    };
    TraceVariable { kind : kind, global_step_count: global_step_count, index: some_index, value: value }
  }
}

pub trait WebAssemblyR1CS<T : Field>{
  fn to_field(value : usize) -> T;
  fn on_const(&mut self, bytes: UntypedValue);
  fn on_br(&mut self, target: Target);
  fn on_set_local(&mut self, local_index : usize);
  fn on_get_local(&mut self, local_index : usize);
  fn on_br_if_eqz(&mut self, target: Target);
  fn on_br_if_nez(&mut self, target: Target);
}

#[allow(non_snake_case)]
impl<T : Field + BigInteger> R1CS<T>
{

  pub const INITIAL_VARIABLE_COUNT : usize = 2;

  pub fn new() -> Self{
    // let mut trace: BTreeMap<VariableIndex, TraceVariable> = BTreeMap::new();
    // insert global variable counter
    // trace.insert(0, TraceVariable{global_step_count : 0, index:0, value: VariableValue::default()});
    // Initialize the proram counter
    // [1, pc,]
    let A : DMatrix<T> = DMatrix::from_vec(1, Self::INITIAL_VARIABLE_COUNT, vec![T::zero(), T::one()]);
    let B : DMatrix<T> = DMatrix::from_vec(1, Self::INITIAL_VARIABLE_COUNT, vec![T::one(), T::zero()]);
    let C : DMatrix<T> = DMatrix::from_vec(1, Self::INITIAL_VARIABLE_COUNT, vec![T::zero(), T::zero()]);

    //


    R1CS::<T>{
      variable_count: Self::INITIAL_VARIABLE_COUNT,
      global_step_count: 0,
      A : A,
      B : B,
      C : C,
      one : TraceVariable {kind:TraceVariableKind::Other, global_step_count:0, index:0,value: T::one()},
      pc : TraceVariable { kind: TraceVariableKind::PC, global_step_count: 0, index: 1, value: T::zero()},
      variables : vec![],
      inputs : vec![],
      // trace : trace,
      // constraints: vec![]
    }
  }

  pub fn init_locals(&mut self, len_locals : usize){
    for i in 0..len_locals{
      let variable : TraceVariable<T> = TraceVariable{
        kind: TraceVariableKind::Other,
        global_step_count : 0, // TODO
        index : self.variable_count,
        value : T::zero()
      };
      self.push_variable(variable);

      // Update the matrices
      let shape = self.A.shape();

      // Modify A
      replace_with_or_abort(&mut self.A, |self_|{
        self_.insert_column(shape.1, T::zero())
      });

      // Modify B
      replace_with_or_abort(&mut self.B, |self_|{
        self_.insert_column(shape.1, T::zero())
      });

      // Modify C
      replace_with_or_abort(&mut self.C, |self_|{
        self_.insert_column(shape.1, T::zero())
      });
    }

  }

  fn push_variable(&mut self, variable: TraceVariable<T>) {
    self.variables.push(variable);
    self.variable_count += 1;
  }

  fn append_row(&mut self){
    let shape = self.A.shape();
    // Modify A
    replace_with_or_abort(&mut self.A, |self_|{
      self_.insert_row(shape.0, T::zero())
    });

    // Modify B
    replace_with_or_abort(&mut self.B, |self_|{
        self_.insert_row(shape.0, T::zero())
    });

    // Modify C
    replace_with_or_abort(&mut self.C, |self_|{
        self_.insert_row(shape.0, T::zero())
    });
  }

  fn append_column(&mut self){
    let shape = self.A.shape();
    // Modify A
    replace_with_or_abort(&mut self.A, |self_|{
      self_.insert_column(shape.1, T::zero())
    });

    // Modify B
    replace_with_or_abort(&mut self.B, |self_|{
        self_.insert_column(shape.1, T::zero())
    });

    // Modify C
    replace_with_or_abort(&mut self.C, |self_|{
        self_.insert_column(shape.1, T::zero())
    });
  }

  pub fn zksnark_spartan(&self){
    let shape = self.A.shape();
    let mut count_A : usize = 0;
    let mut count_B : usize = 0;
    let mut count_C : usize = 0;

    // A
    let mut spartan_A: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let itr_a = self.A.iter();
    for e in itr_a{
      let row_index : usize = count_A / shape.1;
      let col_index : usize = count_A % shape.1;
      if *e != T::zero(){
        spartan_A.push((row_index, col_index, e.to_bytes_le()));
      }
      count_A += 1;
    }

    // B
    let mut spartan_B: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let itr_b = self.A.iter();
    for e in itr_b{
      let row_index : usize = count_B/ shape.1;
      let col_index : usize = count_B % shape.1;
      if *e != T::zero(){
        spartan_B.push((row_index, col_index, e.to_bytes_le()));
      }
      count_B += 1;
    }

    // C
    let mut spartan_C: Vec<(usize, usize, [u8; 32])> = Vec::new();
    let itr_c = self.A.iter();
    for e in itr_c{
      let row_index : usize = count_C / shape.1;
      let col_index : usize = count_C % shape.1;
      if *e != T::zero(){
        spartan_C.push((row_index, col_index, e.to_bytes_le()));
      }
      count_C += 1;
    }
    println!("");
  }

}

impl<T> WebAssemblyR1CS<T> for R1CS<T> where T: Field + BigInteger{

  fn to_field(value : usize) -> T {
    // To be implemented
    T::zero()
  }

  fn on_const(&mut self, bytes: UntypedValue){
    let variable: TraceVariable<T> = TraceVariable {
      kind: TraceVariableKind::Other, 
      global_step_count: 0, // TODO 
      index: self.variable_count, 
      value: Self::to_field(bytes.to_bits() as usize)
    };

    let shape = self.A.shape();
    // Modify A
    replace_with_or_abort(&mut self.A, |self_|{
      self_.insert_column(shape.1, T::zero())
    });
    replace_with_or_abort(&mut self.A, |self_|{
      self_.insert_row(shape.0, T::zero())
    });
    self.A[(shape.0, variable.index)] = T::one();

    // Modify B
    replace_with_or_abort(&mut self.B, |self_|{
      self_.insert_column(shape.1, T::zero())
    });
    replace_with_or_abort(&mut self.B, |self_|{
        self_.insert_row(shape.0, T::zero())
    });
    self.B[(shape.0, 0)] = T::one();

    // Modify C
    replace_with_or_abort(&mut self.C, |self_|{
      self_.insert_column(shape.1, T::zero())
    });
    replace_with_or_abort(&mut self.C, |self_|{
        self_.insert_row(shape.0, T::zero())
    });
    self.C[(shape.0, 0)] = variable.value;    

    // Increment variable count
    self.variable_count += 1;

    self.variables.push(variable);

  }

  fn on_br(&mut self, target : Target){
    // Arithmetization for zkSNARKs
    // let variable =self.r1cs.trace.get_mut(&0).unwrap();
    // TODO: Set variable value
    let pc_variable : TraceVariable<T> = TraceVariable::new(
        TraceVariableKind::PC,
        self.global_step_count, 
        None, 
        Self::to_field(target.destination_pc().into_usize()) // TODO
    );
    let shape = self.A.shape();
    self.global_step_count += 1;

    // Modify A
    replace_with_or_abort(&mut self.A, |self_|{
        self_.insert_row(shape.0, T::zero())
    });
    self.A[(shape.0, pc_variable.index)] = T::one();

    // Modify B
    replace_with_or_abort(&mut self.B, |self_|{
        self_.insert_row(shape.0, T::zero())
    });
    self.B[(shape.0, 0)] = T::one();

    // Modify C
    replace_with_or_abort(&mut self.C, |self_|{
        self_.insert_row(shape.0, T::zero())
    });
    self.C[(shape.0, 0)] = Self::to_field(target.destination_pc().into_usize());    

    println!("r1cs shape: {:?}", &self.A.shape());
    // println!("r1cs: {:?}", &self.r1cs.A);
    // let A = self.r1cs.A.insert_row(num_row.0, VariableValue::zero());
    
    // let term = SparseTerm::new(vec![(0, 1)]);
    // // TODO: get the correct coeffecient
    // let polynomial : SparsePolynomial<VariableValue, SparseTerm> = SparsePolynomial{num_vars:1, terms:vec![(VariableValue::default(), term)]};
    // self.r1cs.constraints.push(polynomial);
  }

  fn on_set_local(&mut self, local_index : usize) {
    // index corresponds to the index in the code instead of wasmi
    let global_index = Self::INITIAL_VARIABLE_COUNT + local_index;
    let variable_last = self.variables.last().unwrap();
    
    // Add constraint
    let shape = self.A.shape();

    // Modify A
    replace_with_or_abort(&mut self.A, |self_|{
      self_.insert_row(shape.0, T::zero())
    });
    self.A[(shape.0, global_index)] = T::one();

    // Modify B
    replace_with_or_abort(&mut self.B, |self_|{
        self_.insert_row(shape.0, T::zero())
    });
    self.B[(shape.0, 0)] = T::one();

    // Modify C
    replace_with_or_abort(&mut self.C, |self_|{
        self_.insert_row(shape.0, T::zero())
    });
    self.C[(shape.0, variable_last.index)] = T::one();
  }

  fn on_get_local(&mut self, local_index : usize) {
    // index corresponds to the index in the code instead of wasmi
    let variable: TraceVariable<T> = TraceVariable {
      kind: TraceVariableKind::Other, 
      global_step_count: 0, // TODO 
      index: self.variable_count, 
      value: self.variables[local_index].value
    };
    self.append_column();
    self.push_variable(variable);
  }

  fn on_br_if_eqz(&mut self, target: Target) {
    
  }

  fn on_br_if_nez(&mut self, target: Target) {
    // Constraint: (target - pc)*z = 0
    self.append_row();
    let shape = self.A.shape();

    let last_variable = self.variables.last().unwrap();
    let target_as_field = Self::to_field(target.destination_pc().into_usize());
    self.A[(shape.0-1, self.one.index)] = target_as_field.neg();
    self.A[(shape.0-1, self.pc.index)] = T::one();

    self.B[(shape.0-1, last_variable.index)] = T::one();


  }

}