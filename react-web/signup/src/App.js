import React from 'react';
import './App.css';
import { makeStyles } from '@material-ui/core/styles';
import TopNavBar from './components/TopNavBar'
import SignupPage from './pages/SignupPage'
import MessageDialog from './components/MessageDialog'
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
  const [dialogTitle, setDialogTitle] = React.useState("");
  const [dialogMessage, setDialogMessage] = React.useState("");
  const [isMessageDialogOpen, setMessageDialogOpen] = React.useState(false);
  
  const handleMessageDialogOpen = () => {
    setMessageDialogOpen(true);
  };

  const handleMessageDialogClose = () => {
    setMessageDialogOpen(false);
  };

  return (
    <div className={classes.root}>
      <MessageDialog
        isOpen={isMessageDialogOpen}
        handleClose={handleMessageDialogClose}
        title={dialogTitle}
        message={dialogMessage}
      ></MessageDialog>
      <TopNavBar 
        userData={userData} 
        setUserData={setUserData}
      />
      <SignupPage
          handleClose={handleMessageDialogClose}
          setDialogTitle={setDialogTitle}
          setDialogMessage={setDialogMessage}
          openMessageDialog={handleMessageDialogOpen}
          userData={userData}
          setUserData={setUserData}
      />;
    </div>
  );
}

export default App;
