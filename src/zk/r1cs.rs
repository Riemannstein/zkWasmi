#![allow(unused)]
use ark_bls12_381::Fq;
// use ark_poly::polynomial::multivariate::{SparsePolynomial, SparseTerm};
use nalgebra::{DMatrix, DVector};
// use ark_poly::MVPolynomial;
use crate::{
    core::UntypedValue,
    engine::{bytecode::Instruction, Target},
};
use ark_ff::fields::Field;
// use ark_ff::BigInteger;
use ark_std::io::Write;
// use bitvec::array::BitArray;
use bitvec::{
    bitarr, bitvec,
    field::BitField,
    prelude::{bits, BitOrder, BitRef, BitSlice, BitStore, BitVec, Lsb0, Msb0},
    view::BitView,
};
use libspartan::{InputsAssignment, Instance, SNARKGens, VarsAssignment, SNARK};
use merlin::Transcript;
use replace_with::replace_with_or_abort;
use std::mem::size_of;
type GlobalStepCount = usize;
type VariableIndex = usize;
pub type VariableValue = Fq;
use std::mem::discriminant;

#[derive(Debug)]
pub struct FieldWriter {
    bytes: Vec<u8>,
}

impl FieldWriter {
    fn bytes(&self) -> &Vec<u8> {
        &self.bytes
    }
}

impl Write for FieldWriter {
    fn write(&mut self, buf: &[u8]) -> alloc::io::Result<usize> {
        self.bytes = buf.to_vec();
        Ok(0)
    }

    fn flush(&mut self) -> alloc::io::Result<()> {
        self.bytes.clear();
        Ok(())
    }
}

#[derive(Debug)]
#[allow(non_snake_case)]
pub struct R1CS<T: Field> {
    variable_count: usize,
    global_step_count: GlobalStepCount,
    A: DMatrix<T>,
    B: DMatrix<T>,
    C: DMatrix<T>,
    one: TraceVariable<T>,
    pc: TraceVariable<T>,
    variable_assignments : DVector<T>, 
    // locals : Vec<TraceVariable<T>>,
    variables: Vec<TraceVariable<T>>,
    inputs: Vec<TraceVariable<T>>,
    stack_variables: Vec<TraceVariable<T>>, // pub trace : BTreeMap<VariableIndex, TraceVariable>,
}

#[derive(Debug, Clone)]
pub enum TraceVariableKind {
    PC,
    Other,
}

#[derive(Debug, Clone)]
pub struct TraceVariable<T: Field> {
    pub kind: TraceVariableKind,
    pub global_step_count: GlobalStepCount,
    pub index: VariableIndex, // global index with respect to the matrix
    pub value: T,
}

impl<T: Field> TraceVariable<T> {
    pub fn new(
        kind: TraceVariableKind,
        global_step_count: GlobalStepCount,
        index: Option<VariableIndex>,
        value: T,
    ) -> Self {
        let some_index: usize = match &kind {
            TraceVariableKind::PC => R1CS::<T>::INITIAL_VARIABLE_COUNT - 1,
            TraceVariableKind::Other => index.unwrap(),
        };
        TraceVariable {
            kind: kind,
            global_step_count: global_step_count,
            index: some_index,
            value: value,
        }
    }
}

pub trait WebAssemblyR1CS<T: Field> {
    fn encode_instruction(instruction: Instruction) -> T;
    fn to_field(value: usize) -> T;
    fn on_const(&mut self, bytes: UntypedValue);
    fn on_br(&mut self, target: Target);
    fn on_set_local(&mut self, local_index: usize);
    fn on_get_local(&mut self, local_index: usize);
    fn on_br_if_eqz(&mut self, target: Target);
    fn on_br_if_nez(&mut self, target: Target);
    fn on_i32_add(&mut self);
    fn on_i32_gt_s(&mut self);
    fn on_i32_gt_u(&mut self);
}

#[allow(non_snake_case)]
impl<T: Field> R1CS<T> {
    pub const INITIAL_VARIABLE_COUNT: usize = 2;

    fn check_new_constraints_sat(&self, start: usize, end: usize) {
        for i in start..end {
            println!("check_new_constraints_sat:\nA.shape(): {:?}\nB.shape(): {:?}\nC.shape(): {:?}\nvariable_assignments len: {:?}", 
            &self.A.shape(), &self.B.shape(), &self.C.shape(), &self.variable_assignments.len());
            let lhs = self.A.row(i)*&self.variable_assignments*(self.B.row(i)*&self.variable_assignments);
            let rhs = self.C.row(i)*&self.variable_assignments;
            assert!(lhs == rhs);
        }
    }

    fn local_to_global_index(&self, local_index: usize) -> usize {
        Self::INITIAL_VARIABLE_COUNT + local_index
    }

    fn serialize_field(field: &T) -> Vec<u8> {
        let mut field_writer = FieldWriter { bytes: vec![] };
        field.serialize(&mut field_writer);
        let bytes = field_writer.bytes().clone();
        field_writer.flush();
        bytes
    }

    fn to_spartan_bytes(field: &T) -> [u8; 32] {
        let byte_vec = Self::serialize_field(field);
        let mut byte_array: [u8; 32] = [0; 32];
        for i in 0..32 {
            if i >= byte_vec.len() {
                break;
            } else {
                byte_array[i] = byte_vec[i];
            }
        }
        byte_array
    }

    pub fn decompose_field(value: T) {
        let modulus = 2;
        let mut even: [u8; 32] = [0; 32];
        let mut odd: [u8; 32] = [0; 32];
        let bytes = Self::serialize_field(&value);
        for i in 0..bytes.len() {
            let bits = (&bytes[i]).view_bits::<Lsb0>().to_bitvec();
            let mut byte_even = bitvec![0, 0, 0, 0, 0, 0, 0, 0];
            let mut byte_odd = bitvec![0, 0, 0, 0, 0, 0, 0, 0];
            for j in 0..bits.len() {
                if (j % 2) == 0 {
                    // Even bit
                    byte_even.set(j, bits[j]);
                } else {
                    // Odd bit
                    byte_odd.set(j, bits[j]);
                }
            }
            even[i] = byte_even.load::<u8>();
            odd[i] = byte_odd.load::<u8>();
        }
        // let event_bits = ;
        // let odd_bis = ;
    }

    pub fn new() -> Self {
        // let mut trace: BTreeMap<VariableIndex, TraceVariable> = BTreeMap::new();
        // insert global variable counter
        // trace.insert(0, TraceVariable{global_step_count : 0, index:0, value: VariableValue::default()});
        // Initialize the proram counter
        // [1, pc,]
        let A: DMatrix<T> =
            DMatrix::from_vec(1, Self::INITIAL_VARIABLE_COUNT, vec![T::one(), T::zero()]);
        let B: DMatrix<T> =
            DMatrix::from_vec(1, Self::INITIAL_VARIABLE_COUNT, vec![T::one(), T::zero()]);
        let C: DMatrix<T> =
            DMatrix::from_vec(1, Self::INITIAL_VARIABLE_COUNT, vec![T::one(), T::zero()]);

        // Initialize variable assignements
        let mut variable_assignements = DVector::from_element(2, T::one());
        variable_assignements[1] = T::zero();

        R1CS::<T> {
            variable_count: Self::INITIAL_VARIABLE_COUNT,
            global_step_count: 0,
            A: A,
            B: B,
            C: C,
            one: TraceVariable {
                kind: TraceVariableKind::Other,
                global_step_count: 0,
                index: 0,
                value: T::one(),
            },
            pc: TraceVariable {
                kind: TraceVariableKind::PC,
                global_step_count: 0,
                index: 1,
                value: T::zero(),
            },
            variables: vec![
                TraceVariable {
                    kind: TraceVariableKind::Other,
                    global_step_count: 0,
                    index: 0,
                    value: T::one(),
                },
                TraceVariable {
                    kind: TraceVariableKind::PC,
                    global_step_count: 0,
                    index: 1,
                    value: T::zero(),
                },
            ],
            variable_assignments : variable_assignements,
            inputs: vec![],
            stack_variables: vec![], // trace : trace,
        }
    }

    pub fn init_locals(&mut self, len_locals: usize) {
        for i in 0..len_locals {
            let variable: TraceVariable<T> = TraceVariable {
                kind: TraceVariableKind::Other,
                global_step_count: 0, // TODO
                index: self.variable_count,
                value: T::zero(),
            };
            self.push_variable(variable);

            // // Update the matrices
            // let shape = self.A.shape();

            // // Modify A
            // replace_with_or_abort(&mut self.A, |self_| self_.insert_column(shape.1, T::zero()));

            // // Modify B
            // replace_with_or_abort(&mut self.B, |self_| self_.insert_column(shape.1, T::zero()));

            // // Modify C
            // replace_with_or_abort(&mut self.C, |self_| self_.insert_column(shape.1, T::zero()));
        }
    }

    fn push_variable(&mut self, variable: TraceVariable<T>) {
        // Push the variable to variables and increment the variable_count
        self.variable_assignments = self.variable_assignments.push(variable.value);
        self.variables.push(variable);
        self.variable_count += 1;

        // Update the matrices
        let shape = self.A.shape();

        assert!(true);
        // Modify A
        replace_with_or_abort(&mut self.A, |self_| self_.insert_column(shape.1, T::zero()));

        // Modify B
        replace_with_or_abort(&mut self.B, |self_| self_.insert_column(shape.1, T::zero()));

        // Modify C
        replace_with_or_abort(&mut self.C, |self_| self_.insert_column(shape.1, T::zero()));
    }

    fn append_row(&mut self) {
        let shape = self.A.shape();
        // Modify A
        replace_with_or_abort(&mut self.A, |self_| self_.insert_row(shape.0, T::zero()));

        // Modify B
        replace_with_or_abort(&mut self.B, |self_| self_.insert_row(shape.0, T::zero()));

        // Modify C
        replace_with_or_abort(&mut self.C, |self_| self_.insert_row(shape.0, T::zero()));
    }

    // fn append_column(&mut self) {
    //     let shape = self.A.shape();
    //     // Modify A
    //     replace_with_or_abort(&mut self.A, |self_| self_.insert_column(shape.1, T::zero()));

    //     // Modify B
    //     replace_with_or_abort(&mut self.B, |self_| self_.insert_column(shape.1, T::zero()));

    //     // Modify C
    //     replace_with_or_abort(&mut self.C, |self_| self_.insert_column(shape.1, T::zero()));
    // }

    pub fn zk_snark_spartan(&mut self) {
        let shape = self.A.shape();
        let mut count_A: usize = 0;
        let mut count_B: usize = 0;
        let mut count_C: usize = 0;

        // A
        let mut spartan_A: Vec<(usize, usize, [u8; 32])> = Vec::new();
        let itr_a = self.A.iter();
        for e in itr_a {
            let row_index: usize = count_A / shape.1;
            let col_index: usize = count_A % shape.1;
            if *e != T::zero() {
                spartan_A.push((row_index, col_index, Self::to_spartan_bytes(e)));
            }
            count_A += 1;
        }

        // B
        let mut spartan_B: Vec<(usize, usize, [u8; 32])> = Vec::new();
        let itr_b = self.B.iter();
        for e in itr_b {
            let row_index: usize = count_B / shape.1;
            let col_index: usize = count_B % shape.1;
            if *e != T::zero() {
                spartan_B.push((row_index, col_index, Self::to_spartan_bytes(e)));
            }
            count_B += 1;
        }

        // C
        let mut spartan_C: Vec<(usize, usize, [u8; 32])> = Vec::new();
        let itr_c = self.C.iter();
        for e in itr_c {
            let row_index: usize = count_C / shape.1;
            let col_index: usize = count_C % shape.1;
            if *e != T::zero() {
                spartan_C.push((row_index, col_index, Self::to_spartan_bytes(e)));
            }
            count_C += 1;
        }

        let num_cons = shape.0; // number of constraints
        let num_vars = self.variables.len() - 1; // Extra two variables are the one and pc
        let num_inputs = self.inputs.len();
        assert!((num_vars + num_inputs) == self.A.shape().1 - 1);
        let inst = Instance::new(
            num_cons, num_vars, num_inputs, &spartan_A, &spartan_B, &spartan_C,
        )
        .unwrap();

        // Create Variable Assignment
        let variables_bytes: Vec<_> = self
            .variables
            .iter()
            .map(|x| Self::to_spartan_bytes(&x.value))
            .collect();
        let assignment_vars = VarsAssignment::new(&variables_bytes).unwrap();

        // Create Input Assignment
        let inputs_bytes: Vec<_> = self
            .inputs
            .iter()
            .map(|x| Self::to_spartan_bytes(&x.value))
            .collect();
        let assignment_inputs = InputsAssignment::new(&inputs_bytes).unwrap();

        // Get number of non-zero entries
        let num_non_zero_entries = num_vars;

        // Check satisfiablility
        let res = inst.is_sat(&assignment_vars, &assignment_inputs);
        assert!(res.unwrap() == true);
        // produce public parameters
        let gens = SNARKGens::new(num_cons, num_vars, num_inputs, num_non_zero_entries);

        // create a commitment to the R1CS instance
        let (comm, decomm) = SNARK::encode(&inst, &gens);

        // produce a proof of satisfiability
        let mut prover_transcript = Transcript::new(b"snark_example");
        let proof = SNARK::prove(
            &inst,
            &comm,
            &decomm,
            assignment_vars,
            &assignment_inputs,
            &gens,
            &mut prover_transcript,
        );

        // verify the proof of satisfiability
        let mut verifier_transcript = Transcript::new(b"snark_example");
        assert!(proof
            .verify(&comm, &assignment_inputs, &mut verifier_transcript, &gens)
            .is_ok());
        println!("proof verification successful!");
        println!("");
    }

    fn multiply(n: usize, field: T) -> T {
        let bits = n.view_bits::<Lsb0>().to_bitvec();
        let two = T::one() + T::one();
        let mut result = T::one();
        for i in 0..bits.len() {
            let bit = bits[i];
            let coefficient = |x: bool| -> T {
                if x {
                    T::one()
                } else {
                    T::zero()
                }
            }(bit);
            result = coefficient * two.pow([2 * i as u64]) + result;
        }
        result
    }
}

impl<T> WebAssemblyR1CS<T> for R1CS<T>
where
    T: Field,
{
    fn encode_instruction(instruction: Instruction) -> T {
        let instruction_value: usize = match instruction {
            Instruction::GetLocal { local_depth } => 2,
            _ => 0,
        };
        Self::multiply(instruction_value, T::one())
    }

    fn on_i32_gt_u(&mut self) {
        let mut is_gt: TraceVariable<T> = TraceVariable {
            kind: TraceVariableKind::Other,
            global_step_count: 0,
            index: self.variable_count,
            value: T::zero(),
        };

        let right = self.stack_variables.pop().unwrap();
        let left = self.stack_variables.pop().unwrap();

        if left.value > right.value {
            is_gt.value = T::one();
        }

        let i32_max_field_value = T::from(u32::MAX);
        let non_determinstic_value = i32_max_field_value * is_gt.value - left.value + right.value;

        let i32_max_variable: TraceVariable<T> = TraceVariable {
            kind: TraceVariableKind::Other,
            global_step_count: 0,
            index: self.variable_count,
            value: i32_max_field_value,
        };
        self.push_variable(i32_max_variable.clone());

        self.push_variable(is_gt.clone());

        let non_deterministc_variable: TraceVariable<T> = TraceVariable {
            kind: TraceVariableKind::Other,
            global_step_count: 0,
            index: self.variable_count,
            value: non_determinstic_value,
        };

        self.push_variable(non_deterministc_variable.clone());

        // self.append_column();
        self.append_row();

        let shape = self.A.shape();
        self.C[(shape.0, left.index)] = T::one();
        self.C[(shape.0, non_deterministc_variable.index)] = T::one();
        self.C[(shape.0, right.index)] = T::one().neg();
    }

    fn on_i32_gt_s(&mut self) {
        let mut is_gt: TraceVariable<T> = TraceVariable {
            kind: TraceVariableKind::Other,
            global_step_count: 0,
            index: self.variable_count,
            value: T::zero(),
        };

        let right = self.stack_variables.pop().unwrap();
        let left = self.stack_variables.pop().unwrap();

        let r1 = &mut is_gt;
        if left.value > right.value {
            r1.value = T::one();
        }

        self.push_variable(is_gt.clone());
        self.append_row();

        let i32_max_field_value = T::from(u32::MAX);

        let i32_max_variable: TraceVariable<T> = TraceVariable {
            kind: TraceVariableKind::Other,
            global_step_count: 0,
            index: self.variable_count,
            value: i32_max_field_value,
        };
        self.push_variable(i32_max_variable.clone());
        self.append_row();

        // let non_determinstic_value = i32_max_field_value * is_gt.value - left.value + right.value;

        // let non_deterministc_variable: TraceVariable<T> = TraceVariable {
        //     kind: TraceVariableKind::Other,
        //     global_step_count: 0,
        //     index: self.variable_count,
        //     value: non_determinstic_value,
        // };

        // self.push_variable(non_deterministc_variable.clone());
        // self.append_row();

        // let shape = self.A.shape();
        // self.C[(shape.0 - 1, left.index)] = T::one();
        // self.C[(shape.0 - 1, non_deterministc_variable.index)] = T::one();
        // self.C[(shape.0 - 1, right.index)] = T::one().neg();
    }

    fn to_field(value: usize) -> T {
        // To be implemented
        T::zero()
    }

    fn on_const(&mut self, bytes: UntypedValue) {
        let variable: TraceVariable<T> = TraceVariable {
            kind: TraceVariableKind::Other,
            global_step_count: 0, // TODO
            index: self.variable_count,
            value: Self::to_field(bytes.to_bits() as usize),
        };
        self.stack_variables.push(variable.clone());

        self.push_variable(variable.clone());
        self.append_row();
        let shape = self.A.shape();
        self.A[(shape.0 - 1, variable.index)] = T::one();
        self.B[(shape.0 - 1, 0)] = T::one();
        self.C[(shape.0 - 1, 0)] = variable.value;
        // self.check_new_constraints_sat(shape.0-1, shape.0);
    }

    fn on_br(&mut self, target: Target) {
        // Arithmetization for zkSNARKs
        // let variable =self.r1cs.trace.get_mut(&0).unwrap();
        // TODO: Set variable value
        let pc_variable: TraceVariable<T> = TraceVariable::new(
            TraceVariableKind::PC,
            self.global_step_count,
            None,
            Self::to_field(target.destination_pc().into_usize()), // TODO
        );
        let shape = self.A.shape();
        self.global_step_count += 1;

        // Modify A
        replace_with_or_abort(&mut self.A, |self_| self_.insert_row(shape.0, T::zero()));
        self.A[(shape.0, pc_variable.index)] = T::one();

        // Modify B
        replace_with_or_abort(&mut self.B, |self_| self_.insert_row(shape.0, T::zero()));
        self.B[(shape.0, 0)] = T::one();

        // Modify C
        replace_with_or_abort(&mut self.C, |self_| self_.insert_row(shape.0, T::zero()));
        self.C[(shape.0, 0)] = Self::to_field(target.destination_pc().into_usize());

        println!("r1cs shape: {:?}", &self.A.shape());
    }

    fn on_set_local(&mut self, local_index: usize) {
        // index corresponds to the index in the code instead of wasmi
        let global_index = Self::INITIAL_VARIABLE_COUNT + local_index;
        let variable_last_index = self.stack_variables.pop().unwrap().index;

        // // Modify A
        // self.append_row();

        // // Add constraint
        // let shape = self.A.shape();
        // self.A[(shape.0 - 1, global_index)] = T::one();

        // // Modify B
        // self.B[(shape.0 - 1, 0)] = T::one();

        // // Modify C
        // self.C[(shape.0 - 1, variable_last_index)] = T::one();
    }

    fn on_get_local(&mut self, local_index: usize) {
        // index corresponds to the index in the code instead of wasmi
        let variable: TraceVariable<T> = TraceVariable {
            kind: TraceVariableKind::Other,
            global_step_count: 0, // TODO
            index: self.variable_count,
            value: self.variables[self.local_to_global_index(local_index)].value,
        };
        self.stack_variables.push(variable.clone());
    }

    fn on_br_if_eqz(&mut self, target: Target) {}

    fn on_br_if_nez(&mut self, target: Target) {
        // Constraint: (target - pc)*z = 0
        self.append_row();
        let shape = self.A.shape();

        let last_variable = self.variables.last().unwrap();
        let target_as_field = Self::to_field(target.destination_pc().into_usize());
        self.A[(shape.0 - 1, self.one.index)] = target_as_field.neg();
        self.A[(shape.0 - 1, self.pc.index)] = T::one();

        self.B[(shape.0 - 1, last_variable.index)] = T::one();
        self.check_new_constraints_sat(shape.0-1, shape.0);
    }

    fn on_i32_add(&mut self) {
        // index corresponds to the index in the code instead of wasmi
        let first_operand = self.stack_variables.pop().unwrap();
        let second_operand = self.stack_variables.pop().unwrap();
        let variable: TraceVariable<T> = TraceVariable {
            kind: TraceVariableKind::Other,
            global_step_count: 0, // TODO
            index: self.variable_count,
            value: first_operand.value + second_operand.value,
        };

        // self.append_column();
        self.append_row();

        self.A[first_operand.index] = T::one();
        self.A[second_operand.index] = T::one();

        self.B[self.one.index] = T::one();

        self.C[variable.index] = T::one();
        self.stack_variables.push(variable.clone());
        self.push_variable(variable);
    }
}
