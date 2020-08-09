import React from 'react';
import './App.css';
import { makeStyles } from '@material-ui/core/styles';
import TopNavBar from './components/TopNavBar'
import LandingPage from './pages/LandingPage'
import { HOME_PAGE, DASHBOARD_PAGE } from './Constants';
import Cookies from 'universal-cookie';

const useStyles = makeStyles((theme) => ({
  root: {
    height: "100%"
  },
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
    userId: cookies.get('userId'),
    userName: cookies.get('userName'),
    userEmail: cookies.get('userEmail'),
  });

  return (
    <div className={classes.root}>
      <TopNavBar 
        userData={userData} 
        setUserData={setUserData}
      />
      <LandingPage
          userData={userData}
      />;
    </div>
  );
}

export default App;
