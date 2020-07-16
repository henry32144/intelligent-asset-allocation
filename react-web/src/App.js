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

const useStyles = makeStyles((theme) => ({
  title: {
    textAlign: 'initial',
    margin: theme.spacing(4, 0, 2),
  },
}));

function App() {
  const classes = useStyles();
  const [userData, setSelectedData] = React.useState({
    userName: "Hello",
    userEmail: "test@gmail.com",
    userPassword: "123"
  });

  return (
    <div className="App">
      <div className={classes.root}>
        <TopNavBar userData={userData}></TopNavBar>
        <PortfolioPage></PortfolioPage>
      </div>
    </div>
  );
}

export default App;
