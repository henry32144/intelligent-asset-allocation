import React from 'react';
import './App.css';
import { makeStyles } from '@material-ui/core/styles';
import TopNavBar from './components/TopNavBar'
import SubmitSelectionButton from './components/SubmitSelectionButton';
import Grid from '@material-ui/core/Grid';
import InvestmentStrategyRadios from './components/InvestmentStrategyRadios'
import Typography from '@material-ui/core/Typography';
import StockSelectSection from './views/StockSelectSection'
import PortfolioPage from './pages/PortfolioPage'
import Cookies from 'universal-cookie';



const useStyles = makeStyles((theme) => ({
  title: {
    textAlign: 'initial',
    margin: theme.spacing(4, 0, 2),
  },
}));

function App() {
  const classes = useStyles();
  // Try to get user data from cookies
  const cookies = new Cookies();
  const [userData, setUserData] = React.useState({
    userName: cookies.get('userName'),
    userEmail: cookies.get('userEmail'),
  });


  return (
    <div className="App">
      <div className={classes.root}>
        <TopNavBar userData={userData} setUserData={setUserData}></TopNavBar>
        <PortfolioPage></PortfolioPage>
      </div>
    </div>
  );
}

export default App;
