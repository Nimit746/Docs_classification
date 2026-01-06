import { Route, Routes } from 'react-router-dom'
import Layout from './pages/Layout'
import Home from './pages/Home'
import NotFound from './pages/NotFound'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout/>} >
        <Route index element={<Home/>} />
        {/* <Route path=" " element={ } /> */}
      </Route>
      <Route path='*' element={<NotFound/>} />
    </Routes>
  )
}
export default App