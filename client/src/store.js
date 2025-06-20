import { configureStore } from '@reduxjs/toolkit';

const dummyReducer = (state = { message: 'hello world' }, action) => {
  return state;
};

export const store = configureStore({
  reducer: {
    chat: dummyReducer, 
  },
});
